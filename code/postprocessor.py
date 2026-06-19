"""
postprocessor.py — Stage 4: Decision Engine.

Applies deterministic rule-based corrections ON TOP of the VLM output:

Rules:
  R1. Merge history_flags from user_history into risk_flags.
  R2. If evidence_standard_met=False → force claim_status=not_enough_information.
  R3. If valid_image=False → force evidence_standard_met=False.
  R4. If claim_status=not_enough_information + evidence_standard_met=True
      → set evidence_standard_met=False (logical consistency).
  R5. Deduplicate and sort risk_flags; remove 'none' when other flags present.
  R6. Serialize lists to the exact CSV format required by the output schema.

These rules catch cases where the VLM is logically inconsistent.
"""

from typing import List

from schemas import ClaimRecord, OutputRow, RiskFlag, UserHistory, VLMOutput


# ── Rule engine ────────────────────────────────────────────────────────────────

def apply_rules(
    vlm_output: VLMOutput,
    record: ClaimRecord,
    user_history: UserHistory,
) -> OutputRow:
    """
    Merge VLM output + user history → final validated OutputRow.

    Args:
        vlm_output:   Raw VLM output (Pydantic-validated).
        record:       The original input ClaimRecord.
        user_history: UserHistory for the claimant.

    Returns:
        OutputRow ready to be written to output.csv.
    """

    # ── R1: Merge history flags ────────────────────────────────────────────────
    history_flags: List[str] = user_history.flag_list()
    vlm_flags: List[str] = [
        f.value for f in vlm_output.risk_flags if f.value != "none"
    ]

    # Combine, preserving valid RiskFlag members only
    all_flags_raw = set(vlm_flags) | set(history_flags)
    # Keep only valid enum values
    valid_flag_values = {f.value for f in RiskFlag}
    merged_flags = sorted(all_flags_raw & valid_flag_values)

    # ── R3: valid_image=False forces evidence_standard_met=False ──────────────
    evidence_standard_met: bool = vlm_output.evidence_standard_met
    if not vlm_output.valid_image:
        evidence_standard_met = False

    # ── R2: evidence_standard_met=False forces not_enough_information ─────────
    claim_status: str = vlm_output.claim_status.value
    if not evidence_standard_met:
        claim_status = "not_enough_information"

    # ── R4: Logical consistency — NEI claim_status → evidence_standard_met=F ──
    if claim_status == "not_enough_information" and evidence_standard_met:
        evidence_standard_met = False

    # ── R5: Clean up risk_flags list ──────────────────────────────────────────
    if not merged_flags:
        risk_flags_str = "none"
    else:
        # Remove stray "none" entries when there are real flags
        cleaned = [f for f in merged_flags if f != "none"]
        risk_flags_str = ";".join(cleaned) if cleaned else "none"

    # ── Serialize supporting_image_ids ────────────────────────────────────────
    supporting_ids = vlm_output.supporting_image_ids
    if not supporting_ids or all(s.lower() == "none" for s in supporting_ids):
        supporting_ids_str = "none"
    else:
        supporting_ids_str = ";".join(supporting_ids)

    # ── Build final OutputRow ──────────────────────────────────────────────────
    return OutputRow(
        user_id=record.user_id,
        image_paths=record.image_paths,
        user_claim=record.user_claim,
        claim_object=record.claim_object,
        evidence_standard_met=str(evidence_standard_met).lower(),
        evidence_standard_met_reason=vlm_output.evidence_standard_met_reason,
        risk_flags=risk_flags_str,
        issue_type=vlm_output.issue_type.value,
        object_part=vlm_output.object_part,
        claim_status=claim_status,
        claim_status_justification=vlm_output.claim_status_justification,
        supporting_image_ids=supporting_ids_str,
        valid_image=str(vlm_output.valid_image).lower(),
        severity=vlm_output.severity.value,
    )


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from schemas import (
        ClaimStatus, IssueType, RiskFlag, Severity, VLMOutput, ClaimRecord, UserHistory
    )

    print("=== Stage 4: Decision Engine Self-Test ===\n")

    # Build a mock ClaimRecord
    mock_record = ClaimRecord(
        user_id="user_005",
        image_paths="images/sample/case_005/img_1.jpg;images/sample/case_005/img_2.jpg",
        user_claim="My bumper has severe damage",
        claim_object="car",
    )

    # Case A: VLM says contradicted, user has history risk
    mock_vlm_a = VLMOutput(
        evidence_standard_met=True,
        evidence_standard_met_reason="Both images show rear bumper clearly.",
        risk_flags=[RiskFlag.claim_mismatch],
        issue_type=IssueType.scratch,
        object_part="rear_bumper",
        claim_status=ClaimStatus.contradicted,
        claim_status_justification="Only minor scratches visible, not severe damage.",
        supporting_image_ids=["img_1"],
        valid_image=True,
        severity=Severity.low,
    )
    history_a = UserHistory(
        user_id="user_005",
        past_claim_count=7,
        accept_claim=2,
        manual_review_claim=2,
        rejected_claim=3,
        last_90_days_claim_count=4,
        history_flags="user_history_risk",
        history_summary="Several exaggerated vehicle damage claims in recent history",
    )
    row_a = apply_rules(mock_vlm_a, mock_record, history_a)
    print("Case A — contradicted + user_history_risk merge:")
    print(f"  claim_status : {row_a.claim_status}")
    print(f"  risk_flags   : {row_a.risk_flags}")
    print(f"  severity     : {row_a.severity}")
    assert row_a.risk_flags == "claim_mismatch;user_history_risk", row_a.risk_flags
    assert row_a.claim_status == "contradicted"
    print("  [PASS]\n")

    # Case B: valid_image=False should cascade to evidence_standard_met=False
    # and claim_status=not_enough_information
    mock_vlm_b = VLMOutput(
        evidence_standard_met=True,   # VLM incorrectly set this to true
        evidence_standard_met_reason="Image shows something.",
        risk_flags=[RiskFlag.wrong_object],
        issue_type=IssueType.unknown,
        object_part="unknown",
        claim_status=ClaimStatus.supported,  # VLM also got this wrong
        claim_status_justification="Image appears to show damage.",
        supporting_image_ids=["img_1"],
        valid_image=False,            # but valid_image is false
        severity=Severity.unknown,
    )
    history_b = UserHistory(
        user_id="user_002",
        past_claim_count=0,
        accept_claim=0,
        manual_review_claim=0,
        rejected_claim=0,
        last_90_days_claim_count=0,
        history_flags="none",
        history_summary="No history.",
    )
    row_b = apply_rules(mock_vlm_b, mock_record, history_b)
    print("Case B — valid_image=False cascades to NEI:")
    print(f"  valid_image          : {row_b.valid_image}")
    print(f"  evidence_standard_met: {row_b.evidence_standard_met}")
    print(f"  claim_status         : {row_b.claim_status}")
    assert row_b.valid_image == "false"
    assert row_b.evidence_standard_met == "false"
    assert row_b.claim_status == "not_enough_information"
    print("  [PASS]\n")

    # Case C: No real flags → "none"
    mock_vlm_c = VLMOutput(
        evidence_standard_met=True,
        evidence_standard_met_reason="Clear image.",
        risk_flags=[],
        issue_type=IssueType.dent,
        object_part="door",
        claim_status=ClaimStatus.supported,
        claim_status_justification="Dent visible.",
        supporting_image_ids=["img_1"],
        valid_image=True,
        severity=Severity.medium,
    )
    history_c = UserHistory(
        user_id="user_001",
        past_claim_count=2,
        accept_claim=2,
        manual_review_claim=0,
        rejected_claim=0,
        last_90_days_claim_count=1,
        history_flags="none",
        history_summary="Clean history.",
    )
    row_c = apply_rules(mock_vlm_c, mock_record, history_c)
    print("Case C — clean claim, no flags:")
    print(f"  risk_flags   : {row_c.risk_flags}")
    assert row_c.risk_flags == "none", row_c.risk_flags
    print("  [PASS]\n")

    print("[DONE] Stage 4 complete -- decision engine all rules pass.")
