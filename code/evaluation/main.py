"""
evaluation/main.py -- Stage 5: Evaluation Framework.

Runs the full pipeline on dataset/sample_claims.csv (which has ground truth)
and computes field-level accuracy metrics.

Features:
  - Checkpoint/resume: saves progress after every successful claim.
    If a run is interrupted by a 429, re-running picks up from the last saved claim.
  - Patience mode: MAX_RETRIES honoured from vlm_analyzer; delay between calls
    is set conservatively to avoid hitting rate limits.

Usage:
    python code/evaluation/main.py
    python code/evaluation/main.py --delay 90   # extra patient (safe for free tier)
    python code/evaluation/main.py --resume     # continue from checkpoint
    python code/evaluation/main.py --fresh      # ignore checkpoint, start over
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVIDENCE_REQUIREMENTS_CSV,
    GEMINI_API_KEY,
    INTER_CALL_DELAY_SECONDS,
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
CHECKPOINT_PATH = EVAL_DIR / "checkpoint.json"

SCOREABLE_FIELDS = [
    "evidence_standard_met",
    "claim_status",
    "issue_type",
    "valid_image",
    "severity",
]


# -- Checkpoint helpers --------------------------------------------------------

def load_checkpoint() -> Dict:
    """Load saved progress. Returns dict with 'results' and 'done_ids'."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Checkpoint loaded: %d claims already done.", len(data["done_ids"]))
        return data
    return {"results": [], "done_ids": []}


def save_checkpoint(results: List[Dict], done_ids: List[str]) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "done_ids": done_ids}, f)


def clear_checkpoint() -> None:
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint cleared -- starting fresh.")


# -- Metrics -------------------------------------------------------------------

def compute_metrics(results: List[Dict]) -> Dict[str, float]:
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

    jaccard_scores = []
    for r in results:
        pred_flags = set(r.get("pred_risk_flags", "").split(";")) - {"", "none"}
        gt_flags = set(r.get("gt_risk_flags", "").split(";")) - {"", "none"}
        if not pred_flags and not gt_flags:
            jaccard_scores.append(1.0)
        elif not pred_flags or not gt_flags:
            jaccard_scores.append(0.0)
        else:
            jaccard_scores.append(len(pred_flags & gt_flags) / len(pred_flags | gt_flags))
    scores["risk_flags_jaccard"] = sum(jaccard_scores) / n

    weights = {
        "claim_status": 3,
        "evidence_standard_met": 2,
        "valid_image": 1,
        "issue_type": 1,
        "severity": 1,
        "risk_flags_jaccard": 1,
    }
    total_w = sum(weights.values())
    scores["composite"] = sum(scores.get(f, 0) * w for f, w in weights.items()) / total_w
    return scores


def print_metrics(scores: Dict[str, float], n: int) -> None:
    print(f"\n{'='*52}")
    print(f"  EVALUATION RESULTS  (n={n})")
    print(f"{'='*52}")
    for field, score in scores.items():
        bar = "#" * int(score * 20)
        print(f"  {field:<28} {score:5.1%}  [{bar:<20}]")
    print(f"{'='*52}\n")


# -- Main evaluation loop ------------------------------------------------------

def run_evaluation(delay: float = INTER_CALL_DELAY_SECONDS,
                   resume: bool = True,
                   fresh: bool = False) -> None:
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY env var not set.")
        sys.exit(1)

    if fresh:
        clear_checkpoint()

    checkpoint = load_checkpoint() if resume else {"results": [], "done_ids": []}
    results: List[Dict] = checkpoint["results"]
    done_ids: List[str] = checkpoint["done_ids"]

    logger.info("Loading datasets...")
    sample_claims = load_claims(SAMPLE_CLAIMS_CSV)
    user_history_map = load_user_history(USER_HISTORY_CSV)
    evidence_reqs_map = load_evidence_requirements(EVIDENCE_REQUIREMENTS_CSV)
    client = build_client()

    pending = [c for c in sample_claims if c.user_id not in done_ids]
    total = len(sample_claims)
    logger.info("Model: %s | Delay: %ss | %d/%d claims remaining",
                __import__("config").GEMINI_MODEL, delay, len(pending), total)

    for claim in pending:
        i = total - len(pending) + pending.index(claim) + 1
        logger.info("[%d/%d] %s | %s", i, total, claim.user_id, claim.claim_object)

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
        done_ids.append(claim.user_id)

        match = output_row.claim_status == (claim.claim_status or "")
        logger.info("  %s claim_status pred=%-26s gt=%s",
                    "[MATCH]" if match else "[DIFF ]",
                    output_row.claim_status, claim.claim_status)

        # Save after every claim so a 429 crash loses nothing
        save_checkpoint(results, done_ids)
        logger.info("  Checkpoint saved (%d/%d done).", len(done_ids), total)

        if pending.index(claim) < len(pending) - 1:
            time.sleep(delay)

    # -- Final output ----------------------------------------------------------
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved evaluation comparison to %s", RESULTS_CSV)

    scores = compute_metrics(results)
    print_metrics(scores, len(results))

    _write_report(scores, total, len(results))
    logger.info("Evaluation report written to %s", REPORT_PATH)

    # Clear checkpoint on clean completion
    clear_checkpoint()


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
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate claim pipeline on sample_claims.csv")
    parser.add_argument("--delay", type=float, default=INTER_CALL_DELAY_SECONDS)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoint if available (default: on)")
    parser.add_argument("--fresh", action="store_true", default=False,
                        help="Ignore checkpoint and start from scratch")
    args = parser.parse_args()
    run_evaluation(delay=args.delay, resume=args.resume, fresh=args.fresh)
