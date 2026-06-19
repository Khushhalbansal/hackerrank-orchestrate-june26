"""
schemas.py — Pydantic models for validated, typed claim outputs.

Using strict enums ensures the VLM never produces off-schema values.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, field_validator


# ── Allowed enum values (copied exactly from problem_statement.md) ─────────────

class ClaimStatus(str, Enum):
    supported = "supported"
    contradicted = "contradicted"
    not_enough_information = "not_enough_information"


class IssueType(str, Enum):
    dent = "dent"
    scratch = "scratch"
    crack = "crack"
    glass_shatter = "glass_shatter"
    broken_part = "broken_part"
    missing_part = "missing_part"
    torn_packaging = "torn_packaging"
    crushed_packaging = "crushed_packaging"
    water_damage = "water_damage"
    stain = "stain"
    none = "none"
    unknown = "unknown"


class Severity(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"
    unknown = "unknown"


class RiskFlag(str, Enum):
    none = "none"
    blurry_image = "blurry_image"
    cropped_or_obstructed = "cropped_or_obstructed"
    low_light_or_glare = "low_light_or_glare"
    wrong_angle = "wrong_angle"
    wrong_object = "wrong_object"
    wrong_object_part = "wrong_object_part"
    damage_not_visible = "damage_not_visible"
    claim_mismatch = "claim_mismatch"
    possible_manipulation = "possible_manipulation"
    non_original_image = "non_original_image"
    text_instruction_present = "text_instruction_present"
    user_history_risk = "user_history_risk"
    manual_review_required = "manual_review_required"


# ── VLM structured output (what Gemini returns as JSON) ───────────────────────

class VLMOutput(BaseModel):
    """
    The structured JSON response expected from Gemini.
    Pydantic validates every field before it reaches the post-processor.
    """
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: List[RiskFlag]
    issue_type: IssueType
    object_part: str
    claim_status: ClaimStatus
    claim_status_justification: str
    supporting_image_ids: List[str]  # image stem names, e.g. ["img_1", "img_2"]
    valid_image: bool
    severity: Severity

    @field_validator("risk_flags", mode="before")
    @classmethod
    def coerce_risk_flags(cls, v):
        """Accept a single string 'none' or a list."""
        if isinstance(v, str):
            v = [item.strip() for item in v.split(";") if item.strip()]
        return v

    @field_validator("supporting_image_ids", mode="before")
    @classmethod
    def coerce_image_ids(cls, v):
        """Accept semicolon-delimited string or list."""
        if isinstance(v, str):
            if v.lower() == "none":
                return []
            v = [item.strip() for item in v.split(";") if item.strip()]
        return v


# ── Flat output row (what goes into output.csv) ───────────────────────────────

class OutputRow(BaseModel):
    """One row in output.csv, with all values serialised to strings."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str        # "true" / "false"
    evidence_standard_met_reason: str
    risk_flags: str                   # semicolon-joined or "none"
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str         # semicolon-joined or "none"
    valid_image: str                  # "true" / "false"
    severity: str


# ── Input record ──────────────────────────────────────────────────────────────

class ClaimRecord(BaseModel):
    """One input row from claims.csv or sample_claims.csv."""
    user_id: str
    image_paths: str          # raw semicolon-delimited string from CSV
    user_claim: str
    claim_object: str
    # Ground-truth fields (present in sample_claims.csv, absent in claims.csv)
    evidence_standard_met: Optional[str] = None
    evidence_standard_met_reason: Optional[str] = None
    risk_flags: Optional[str] = None
    issue_type: Optional[str] = None
    object_part: Optional[str] = None
    claim_status: Optional[str] = None
    claim_status_justification: Optional[str] = None
    supporting_image_ids: Optional[str] = None
    valid_image: Optional[str] = None
    severity: Optional[str] = None

    def get_image_path_list(self) -> List[str]:
        """Return individual image path strings from the semicolon-delimited field."""
        return [p.strip() for p in self.image_paths.split(";") if p.strip()]

    def get_image_ids(self) -> List[str]:
        """Return image stem IDs (filename without extension)."""
        from pathlib import Path
        return [Path(p).stem for p in self.get_image_path_list()]


# ── User history record ───────────────────────────────────────────────────────

class UserHistory(BaseModel):
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str    # "none" or semicolon-delimited flags
    history_summary: str

    def flag_list(self) -> List[str]:
        if self.history_flags.lower() == "none":
            return []
        return [f.strip() for f in self.history_flags.split(";") if f.strip()]
