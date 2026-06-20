"""
Emergency output generator for claims.csv.

Strategy:
  1. Try to call Gemini for as many claims as quota allows.
  2. For any claim that fails (429), apply smart rule-based defaults
     based on user_history risk and evidence_requirements.
  3. Always writes a complete output.csv -- never leaves rows missing.

Run: python code/emergency_output.py
"""
import csv
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLAIMS_CSV, EVIDENCE_REQUIREMENTS_CSV,
    GEMINI_API_KEY, INTER_CALL_DELAY_SECONDS,
    OUTPUT_COLUMNS, OUTPUT_CSV, USER_HISTORY_CSV,
)
from postprocessor import apply_rules
from preprocessor import (
    detect_prompt_injection, get_evidence_requirements, get_user_history,
    load_claims, load_evidence_requirements, load_user_history, resolve_image_paths,
)
from prompt_templates import build_prompt
from schemas import (
    ClaimRecord, ClaimStatus, IssueType, OutputRow, RiskFlag, Severity,
    UserHistory, VLMOutput,
)
from vlm_analyzer import analyze_claim, build_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHECKPOINT = Path(__file__).parent.parent / "output_checkpoint.json"


def smart_fallback(claim: ClaimRecord, uh: UserHistory) -> VLMOutput:
    """
    Rule-based fallback when Gemini is unavailable.
    Uses user_history risk flags and image presence to make a conservative decision.
    """
    import json
    history_flags = uh.flag_list()
    image_ids = claim.get_image_ids()
    image_paths = resolve_image_paths(claim)
    images_exist = any(p.exists() for p in image_paths)

    # High-risk user with missing images -> NEI
    if not images_exist:
        return VLMOutput(
            evidence_standard_met=False,
            evidence_standard_met_reason="No images could be loaded for assessment.",
            risk_flags=[RiskFlag.damage_not_visible],
            issue_type=IssueType.unknown,
            object_part="unknown",
            claim_status=ClaimStatus.not_enough_information,
            claim_status_justification="Images not available for automated review.",
            supporting_image_ids=[],
            valid_image=False,
            severity=Severity.unknown,
        )

    # Has user_history_risk -> flag but still evaluate as supported (conservative)
    risk_flags = []
    if "user_history_risk" in history_flags:
        risk_flags.append(RiskFlag.user_history_risk)
        risk_flags.append(RiskFlag.manual_review_required)

    # Check for injection in claim text
    inj_found, _ = detect_prompt_injection(claim.user_claim)
    if inj_found:
        risk_flags.append(RiskFlag.text_instruction_present)
        risk_flags.append(RiskFlag.manual_review_required)

    return VLMOutput(
        evidence_standard_met=True,
        evidence_standard_met_reason="Image set present and loaded; automated VLM assessment unavailable (API quota).",
        risk_flags=risk_flags if risk_flags else [],
        issue_type=IssueType.unknown,
        object_part="unknown",
        claim_status=ClaimStatus.not_enough_information,
        claim_status_justification="Automated visual assessment could not be completed. Manual review required.",
        supporting_image_ids=image_ids[:1] if image_ids else [],
        valid_image=True,
        severity=Severity.unknown,
    )


def run(delay: float = 7.0) -> None:
    import json

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set")
        sys.exit(1)

    # Load checkpoint (resume from last run)
    done_ids = []
    output_rows = []
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            ckpt = json.load(f)
        done_ids = ckpt.get("done_ids", [])
        output_rows = ckpt.get("output_rows", [])
        logger.info("Checkpoint: %d/%d already done.", len(done_ids), 44)

    claims = load_claims(CLAIMS_CSV)
    user_history_map = load_user_history(USER_HISTORY_CSV)
    evidence_reqs_map = load_evidence_requirements(EVIDENCE_REQUIREMENTS_CSV)
    pending = [c for c in claims if c.user_id not in done_ids]

    quota_exhausted = False
    client = None
    if not quota_exhausted:
        try:
            client = build_client()
        except Exception as e:
            logger.warning("Could not build client: %s", e)

    for i, claim in enumerate(pending):
        idx = len(done_ids) + i + 1
        logger.info("[%d/44] %s | %s", idx, claim.user_id, claim.claim_object)

        uh = get_user_history(claim.user_id, user_history_map)
        reqs = get_evidence_requirements(claim.claim_object, evidence_reqs_map)
        inj_found, _ = detect_prompt_injection(claim.user_claim)
        image_ids = claim.get_image_ids()
        image_paths = resolve_image_paths(claim)

        vlm_output = None

        if not quota_exhausted and client:
            prompt = build_prompt(
                user_claim=claim.user_claim,
                claim_object=claim.claim_object,
                image_ids=image_ids,
                user_history=uh,
                evidence_requirements=reqs,
                injection_detected=inj_found,
            )
            try:
                vlm_output = analyze_claim(prompt, image_paths, client, image_ids)
                logger.info("  [VLM] status=%s", vlm_output.claim_status)
            except RuntimeError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning("  [QUOTA] Quota exhausted -- switching to fallback for remaining claims.")
                    quota_exhausted = True
                else:
                    logger.warning("  [ERROR] %s -- using fallback.", e)

        if vlm_output is None:
            vlm_output = smart_fallback(claim, uh)
            logger.info("  [FALLBACK] status=%s", vlm_output.claim_status)

        output_row = apply_rules(vlm_output, claim, uh)
        output_rows.append(output_row.model_dump())
        done_ids.append(claim.user_id)

        # Save checkpoint after every claim
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump({"done_ids": done_ids, "output_rows": output_rows}, f)

        if i < len(pending) - 1 and not quota_exhausted:
            time.sleep(delay)

    # Write final output.csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    api_count = sum(1 for r in output_rows if r.get("claim_status_justification", "").find("quota") == -1
                    and r.get("claim_status_justification", "").find("Manual review") == -1)
    fallback_count = len(output_rows) - api_count
    logger.info("Done. output.csv written: %d VLM + %d fallback rows.", api_count, fallback_count)
    logger.info("Output: %s", OUTPUT_CSV)


if __name__ == "__main__":
    run(delay=float(os.environ.get("GEMINI_DELAY", "7")))
