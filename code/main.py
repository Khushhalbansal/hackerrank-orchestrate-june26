"""
main.py — Stage 6: Output Generation Entry Point.

Runs the full damage claim verification pipeline on dataset/claims.csv
and writes predictions to output.csv at the repository root.

Usage:
    # Set API key first:
    set GEMINI_API_KEY=your_key_here      (Windows)
    export GEMINI_API_KEY=your_key_here   (macOS/Linux)

    # Run on test claims:
    python code/main.py

    # Run on sample claims (for debugging):
    python code/main.py --input dataset/sample_claims.csv --output output_sample.csv

    # Adjust inter-call delay (seconds):
    python code/main.py --delay 6
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
# Allow running from repo root OR from code/
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from config import (
    CLAIMS_CSV,
    EVIDENCE_REQUIREMENTS_CSV,
    GEMINI_API_KEY,
    INTER_CALL_DELAY_SECONDS,
    OUTPUT_COLUMNS,
    OUTPUT_CSV,
    REPO_ROOT,
    USER_HISTORY_CSV,
)
from postprocessor import apply_rules
from preprocessor import (
    detect_prompt_injection,
    get_evidence_requirements,
    get_user_history,
    load_claims,
    load_evidence_requirements,
    load_user_history,
    resolve_image_paths,
)
from prompt_templates import build_prompt
from vlm_analyzer import analyze_claim, build_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(input_path: Path, output_path: Path, delay: float) -> None:
    """
    Full pipeline: load → prompt → VLM → post-process → write CSV.
    """
    if not GEMINI_API_KEY:
        logger.error(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Run: set GEMINI_API_KEY=your_key_here"
        )
        sys.exit(1)

    # ── Load all data ──────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    claims = load_claims(input_path)
    user_history_map = load_user_history(USER_HISTORY_CSV)
    evidence_reqs_map = load_evidence_requirements(EVIDENCE_REQUIREMENTS_CSV)
    client = build_client()

    logger.info("Processing %d claims. Output -> %s", len(claims), output_path)
    logger.info("Inter-call delay: %.1fs | Model: gemini-2.5-flash", delay)

    output_rows = []

    for i, claim in enumerate(claims):
        logger.info(
            "[%d/%d] user=%s  object=%s  images=%s",
            i + 1, len(claims),
            claim.user_id,
            claim.claim_object,
            claim.get_image_ids(),
        )

        # ── Pre-process ────────────────────────────────────────────────────────
        uh = get_user_history(claim.user_id, user_history_map)
        reqs = get_evidence_requirements(claim.claim_object, evidence_reqs_map)
        inj_found, inj_trigger = detect_prompt_injection(claim.user_claim)
        if inj_found:
            logger.warning("  [INJECTION] Trigger detected: '%s'", inj_trigger)

        image_ids = claim.get_image_ids()
        image_paths = resolve_image_paths(claim)

        missing = [p for p in image_paths if not p.exists()]
        if missing:
            logger.warning("  Missing images: %s", [str(m) for m in missing])

        # ── Build prompt ───────────────────────────────────────────────────────
        prompt = build_prompt(
            user_claim=claim.user_claim,
            claim_object=claim.claim_object,
            image_ids=image_ids,
            user_history=uh,
            evidence_requirements=reqs,
            injection_detected=inj_found,
        )

        # ── VLM call ───────────────────────────────────────────────────────────
        vlm_output = analyze_claim(prompt, image_paths, client, image_ids)

        # ── Post-process / rule engine ─────────────────────────────────────────
        output_row = apply_rules(vlm_output, claim, uh)

        logger.info(
            "  -> status=%-25s evidence_met=%-5s severity=%s",
            output_row.claim_status,
            output_row.evidence_standard_met,
            output_row.severity,
        )

        output_rows.append(output_row.model_dump())

        # Rate-limit: sleep between calls (except after the last one)
        if i < len(claims) - 1:
            time.sleep(delay)

    # ── Write output.csv ───────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info("Done. Wrote %d rows to %s", len(output_rows), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Damage claim verification pipeline"
    )
    parser.add_argument(
        "--input", type=Path, default=CLAIMS_CSV,
        help="Input CSV (default: dataset/claims.csv)",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_CSV,
        help="Output CSV path (default: output.csv at repo root)",
    )
    parser.add_argument(
        "--delay", type=float, default=INTER_CALL_DELAY_SECONDS,
        help="Seconds between API calls (default: %(default)s)",
    )
    args = parser.parse_args()
    run(args.input, args.output, args.delay)


if __name__ == "__main__":
    main()
