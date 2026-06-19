"""
config.py -- Central configuration for the claim verification pipeline.

All secrets are read from environment variables. Never hardcode API keys.
"""

import os
from pathlib import Path

# -- Repository root -----------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()
DATASET_DIR = REPO_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"

# -- Input files ---------------------------------------------------------------
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
CLAIMS_CSV = DATASET_DIR / "claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"

# -- Output --------------------------------------------------------------------
OUTPUT_CSV = REPO_ROOT / "output.csv"

# -- Gemini settings -----------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Override model via env var for flexibility:
#   set GEMINI_MODEL=gemini-1.5-flash   (free tier 1500 RPD, use for evaluation)
#   set GEMINI_MODEL=gemini-2.5-flash   (free tier 20 RPD, use for final output)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# -- Rate-limit / retry settings -----------------------------------------------
# gemini-2.5-flash free tier: 10 RPM, 20 RPD. 7s delay stays under 10 RPM.
# gemini-1.5-flash free tier: 15 RPM, 1500 RPD. 5s delay is sufficient.
INTER_CALL_DELAY_SECONDS = int(os.environ.get("GEMINI_DELAY", "7"))
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 10  # exponential backoff base

# -- Output column order (must match problem_statement.md exactly) -------------
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]
