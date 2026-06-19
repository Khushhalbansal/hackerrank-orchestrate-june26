"""
evaluation/main.py — Stage 5: Evaluation Framework.

Runs the full pipeline on dataset/sample_claims.csv (which has ground truth)
and computes field-level accuracy metrics.

Usage:
    python code/evaluation/main.py
    python code/evaluation/main.py --delay 6    # seconds between API calls
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Make the code/ directory importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVIDENCE_REQUIREMENTS_CSV,
    GEMINI_API_KEY,
    INTER_CALL_DELAY_SECONDS,
    OUTPUT_COLUMNS,
    REPO_ROOT,
    SAMPLE_CLAIMS_CSV,
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

EVAL_DIR = Path(__file__).parent
RESULTS_CSV = EVAL_DIR / "evaluation_results.csv"
REPORT_PATH = EVAL_DIR / "evaluation_report.md"

# Fields we can score against ground truth
SCOREABLE_FIELDS = [
    "evidence_standard_met",
    "claim_status",
    "issue_type",
    "valid_image",
    "severity",
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Compute per-field exact-match accuracy.
    Also computes risk_flag overlap (Jaccard) and a composite score.
    """
    if not results:
        return {}

    n = len(results)
    scores: Dict[str, float] = {}

    for field in SCOREABLE_FIELDS:
        correct = sum(
            1 for r in results
            if r.get(f"pred_{field}", "").strip().lower()
            == r.get(f"gt_{field}", "").strip().lower()
        )
        scores[field] = correct / n

    # Risk flag overlap: Jaccard similarity per row, averaged
    jaccard_scores = []
    for r in results:
        pred_flags = set(r.get("pred_risk_flags", "").split(";")) - {"", "none"}
        gt_flags = set(r.get("gt_risk_flags", "").split(";")) - {"", "none"}
        if not pred_flags and not gt_flags:
            jaccard_scores.append(1.0)
        elif not pred_flags or not gt_flags:
            jaccard_scores.append(0.0)
        else:
            inter = len(pred_flags & gt_flags)
            union = len(pred_flags | gt_flags)
            jaccard_scores.append(inter / union)
    scores["risk_flags_jaccard"] = sum(jaccard_scores) / n

    # Overall composite (weighted)
    weights = {
        "claim_status": 3,
        "evidence_standard_met": 2,
        "valid_image": 1,
        "issue_type": 1,
        "severity": 1,
        "risk_flags_jaccard": 1,
    }
    total_w = sum(weights.values())
    composite = sum(scores.get(f, 0) * w for f, w in weights.items()) / total_w
    scores["composite"] = composite

    return scores


def print_metrics(scores: Dict[str, float], n: int) -> None:
    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS  (n={n})")
    print(f"{'='*50}")
    for field, score in scores.items():
        bar = "#" * int(score * 20)
        print(f"  {field:<28} {score:5.1%}  [{bar:<20}]")
    print(f"{'='*50}\n")


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(delay: float = INTER_CALL_DELAY_SECONDS) -> None:
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY env var not set.")
        sys.exit(1)

    logger.info("Loading datasets...")
    sample_claims = load_claims(SAMPLE_CLAIMS_CSV)
    user_history_map = load_user_history(USER_HISTORY_CSV)
    evidence_reqs_map = load_evidence_requirements(EVIDENCE_REQUIREMENTS_CSV)
    client = build_client()

    results: List[Dict] = []
    predictions: List[Dict] = []

    logger.info("Running pipeline on %d sample claims...", len(sample_claims))

    for i, claim in enumerate(sample_claims):
        logger.info("[%d/%d] %s | %s", i + 1, len(sample_claims),
                    claim.user_id, claim.claim_object)

        uh = get_user_history(claim.user_id, user_history_map)
        reqs = get_evidence_requirements(claim.claim_object, evidence_reqs_map)
        inj_found, _ = detect_prompt_injection(claim.user_claim)
        image_ids = claim.get_image_ids()
        image_paths = resolve_image_paths(claim)

        prompt = build_prompt(
            user_claim=claim.user_claim,
            claim_object=claim.claim_object,
            image_ids=image_ids,
            user_history=uh,
            evidence_requirements=reqs,
            injection_detected=inj_found,
        )

        vlm_output = analyze_claim(prompt, image_paths, client, image_ids)
        output_row = apply_rules(vlm_output, claim, uh)
        pred = output_row.model_dump()
        predictions.append(pred)

        # Build comparison record
        result = {
            "user_id": claim.user_id,
            "claim_object": claim.claim_object,
            "pred_claim_status": output_row.claim_status,
            "gt_claim_status": claim.claim_status or "",
            "pred_evidence_standard_met": output_row.evidence_standard_met,
            "gt_evidence_standard_met": claim.evidence_standard_met or "",
            "pred_issue_type": output_row.issue_type,
            "gt_issue_type": claim.issue_type or "",
            "pred_valid_image": output_row.valid_image,
            "gt_valid_image": claim.valid_image or "",
            "pred_severity": output_row.severity,
            "gt_severity": claim.severity or "",
            "pred_risk_flags": output_row.risk_flags,
            "gt_risk_flags": claim.risk_flags or "",
        }
        results.append(result)

        # Show live diff
        match = output_row.claim_status == (claim.claim_status or "")
        status = "[MATCH]" if match else "[DIFF ]"
        logger.info("  %s claim_status pred=%s  gt=%s",
                    status, output_row.claim_status, claim.claim_status)

        if i < len(sample_claims) - 1:
            time.sleep(delay)

    # ── Save results CSV ───────────────────────────────────────────────────────
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved evaluation comparison to %s", RESULTS_CSV)

    # ── Compute and display metrics ────────────────────────────────────────────
    scores = compute_metrics(results)
    print_metrics(scores, len(results))

    # ── Write evaluation report ────────────────────────────────────────────────
    _write_report(scores, len(sample_claims), len(results))
    logger.info("Evaluation report written to %s", REPORT_PATH)


def _write_report(scores: Dict[str, float], n_claims: int, n_evaluated: int) -> None:
    lines = [
        "# Evaluation Report\n",
        f"Evaluated on `dataset/sample_claims.csv` ({n_evaluated}/{n_claims} claims)\n",
        "\n## Accuracy Metrics\n",
        "| Field | Accuracy |",
        "|---|---|",
    ]
    for field, score in scores.items():
        lines.append(f"| `{field}` | {score:.1%} |")

    lines += [
        "\n## Operational Analysis\n",
        f"- **Model calls**: {n_claims} for sample, ~44 for test set",
        f"- **Images per call**: ~1-3 (avg. ~2)",
        f"- **Delay between calls**: {INTER_CALL_DELAY_SECONDS}s (stays under 10 RPM free-tier limit)",
        f"- **Estimated runtime** (sample): ~{n_claims * INTER_CALL_DELAY_SECONDS // 60}m {n_claims * INTER_CALL_DELAY_SECONDS % 60}s",
        f"- **Estimated runtime** (test, 44 claims): ~{44 * INTER_CALL_DELAY_SECONDS // 60}m {44 * INTER_CALL_DELAY_SECONDS % 60}s",
        "",
        "### Token estimates (Gemini 2.5 Flash)",
        "- Input tokens per call: ~1,500 text + ~500 per image = ~2,500 avg",
        "- Output tokens per call: ~300 (JSON response)",
        "- Total input tokens (44 test): ~110,000",
        "- Total output tokens (44 test): ~13,200",
        "",
        "### Cost estimate (Gemini 2.5 Flash pricing)",
        "- Input: $0.075 / 1M tokens -> 110k tokens ~$0.008",
        "- Output: $0.30 / 1M tokens -> 13k tokens ~$0.004",
        "- **Total estimated cost: < $0.02 for the full test set**",
        "",
        "### Rate limit strategy",
        "- Free tier: 10 RPM, 250k TPM",
        f"- Inter-call delay: {INTER_CALL_DELAY_SECONDS}s -> ~10 RPM max",
        "- Retry: 3 attempts with exponential backoff (10s, 20s, 40s)",
        "- No caching needed at this scale; all 44 claims processed sequentially",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate claim pipeline on sample_claims.csv")
    parser.add_argument("--delay", type=float, default=INTER_CALL_DELAY_SECONDS,
                        help="Seconds between API calls (default: %(default)s)")
    args = parser.parse_args()
    run_evaluation(delay=args.delay)
