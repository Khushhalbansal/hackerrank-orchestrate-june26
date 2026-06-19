"""
vlm_analyzer.py -- Stage 3: Vision Analyzer.

Sends each claim (text prompt + images) to Gemini 2.5 Flash and
returns a validated VLMOutput object.

Key design decisions:
  - All images for a claim are sent in ONE API call (minimises RPM usage).
  - JSON is requested via response_mime_type, not parsed from markdown.
  - Exponential-backoff retry on transient errors (429, 5xx).
  - 429 retryDelay is parsed from the error body and honoured.
  - Pydantic validation ensures no off-schema value ever propagates.
"""

import json
import re
import time
import logging
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_RETRIES, RETRY_BASE_DELAY_SECONDS
from schemas import VLMOutput

logger = logging.getLogger(__name__)


# ── Client factory ─────────────────────────────────────────────────────────────

def build_client() -> genai.Client:
    """Create and return a Gemini API client."""
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Export it before running: set GEMINI_API_KEY=your_key"
        )
    return genai.Client(api_key=GEMINI_API_KEY)


# ── Image loader ───────────────────────────────────────────────────────────────

def _load_image_part(image_path: Path) -> Optional[types.Part]:
    """
    Load a single image from disk and return a Gemini Part object.
    Returns None if the file does not exist (logs a warning).
    """
    if not image_path.exists():
        logger.warning("Image not found, skipping: %s", image_path)
        return None

    suffix = image_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    with open(image_path, "rb") as fh:
        data = fh.read()

    return types.Part.from_bytes(data=data, mime_type=mime_type)


def _parse_retry_delay(error_str: str) -> Optional[int]:
    """
    Extract the suggested retry delay (seconds) from a 429 error message.
    Gemini returns something like: 'retryDelay': '43s' or 'retry in 43.9s'
    """
    m = re.search(r"retry[_\s]?(?:in|delay)[^\d]*(\d+)", error_str, re.IGNORECASE)
    if m:
        return min(int(m.group(1)) + 5, 120)  # honour the hint, cap at 2 min
    return None


# ── Core VLM call ──────────────────────────────────────────────────────────────

def analyze_claim(
    prompt: str,
    image_paths: List[Path],
    client: genai.Client,
    image_ids: Optional[List[str]] = None,
) -> VLMOutput:
    """
    Call Gemini with the claim prompt and all associated images.

    Args:
        prompt:      The full text prompt from prompt_templates.build_prompt().
        image_paths: Absolute paths to the image files.
        client:      Authenticated Gemini client.
        image_ids:   Human-readable IDs (for logging). Optional.

    Returns:
        Validated VLMOutput instance.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    ids_str = str(image_ids or [p.name for p in image_paths])

    # ── Build content parts: images first, then the text prompt ────────────────
    parts: List[types.Part] = []
    loaded_count = 0
    for path in image_paths:
        part = _load_image_part(path)
        if part is not None:
            parts.append(part)
            loaded_count += 1

    if loaded_count == 0:
        logger.warning("No images could be loaded for claim (ids=%s). "
                       "Returning NEI fallback.", ids_str)
        return _fallback_output("No images could be loaded for this claim.")

    parts.append(types.Part.from_text(text=prompt))

    # ── Retry loop ─────────────────────────────────────────────────────────────
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,   # deterministic output
                ),
            )
            raw_text = response.text.strip()
            return _parse_and_validate(raw_text, ids_str)

        except Exception as exc:
            last_error = exc
            exc_str = str(exc)
            # Honour retryDelay hint from 429 body if present
            retry_hint = _parse_retry_delay(exc_str)
            wait = retry_hint if retry_hint else RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed for claim (ids=%s): %s. Retrying in %ds.",
                attempt, MAX_RETRIES, ids_str, exc_str[:120], wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"All {MAX_RETRIES} attempts failed for claim (ids={ids_str}). "
        f"Last error: {last_error}"
    )


# ── JSON parsing & validation ──────────────────────────────────────────────────

def _parse_and_validate(raw_text: str, ids_str: str) -> VLMOutput:
    """
    Parse Gemini's JSON response and validate it against VLMOutput.
    Falls back to a safe NEI output on any parsing error.
    """
    # Strip markdown fences if Gemini adds them despite response_mime_type
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(raw_text)
        return VLMOutput(**data)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error(
            "JSON parse/validation error for claim (ids=%s): %s\nRaw: %s",
            ids_str, exc, raw_text[:500],
        )
        return _fallback_output(f"Response parsing failed: {exc}")


def _fallback_output(reason: str) -> VLMOutput:
    """Safe fallback when the VLM call fails entirely."""
    return VLMOutput(
        evidence_standard_met=False,
        evidence_standard_met_reason=reason,
        risk_flags=[],
        issue_type="unknown",
        object_part="unknown",
        claim_status="not_enough_information",
        claim_status_justification=reason,
        supporting_image_ids=[],
        valid_image=False,
        severity="unknown",
    )


# ── Self-test (runs ONE real Gemini call on sample case_001) ───────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, ".")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from config import SAMPLE_CLAIMS_CSV, GEMINI_API_KEY
    from preprocessor import (
        load_claims, load_user_history, load_evidence_requirements,
        get_user_history, get_evidence_requirements,
        resolve_image_paths, detect_prompt_injection,
    )
    from prompt_templates import build_prompt

    if not GEMINI_API_KEY:
        print("Set GEMINI_API_KEY env var")
        sys.exit(1)

    print("=== Stage 3: Vision Analyzer Self-Test ===")
    print(f"Model: {GEMINI_MODEL}\n")

    claims = load_claims(SAMPLE_CLAIMS_CSV)
    user_history_map = load_user_history()
    evidence_reqs_map = load_evidence_requirements()
    client = build_client()

    claim = claims[0]
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

    print(f"Calling Gemini for {claim.user_id} | {claim.claim_object} | "
          f"images: {image_ids} ...")
    result = analyze_claim(prompt, image_paths, client, image_ids)
    print("\n--- VLM Output ---")
    print(result.model_dump_json(indent=2))

    print("\n--- Ground Truth ---")
    print(f"  claim_status : {claim.claim_status}")
    print(f"  evidence_met : {claim.evidence_standard_met}")
    print(f"  issue_type   : {claim.issue_type}")
    print(f"  severity     : {claim.severity}")

    print("\n[DONE] Stage 3 complete -- VLM analyzer operational.")
