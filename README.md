# HackerRank Orchestrate — Multi-Modal Evidence Review

Automated damage claim verification system using Gemini 2.5 Flash (VLM) with a rule-based postprocessor.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Khushhalbansal/hackerrank-orchestrate-june26
cd hackerrank-orchestrate-june26

# 2. Install dependencies
pip install google-genai pandas pillow tqdm pydantic

# 3. Set API key
set GEMINI_API_KEY=your_gemini_api_key_here   # Windows
# export GEMINI_API_KEY=your_key              # macOS/Linux

# 4. Run on test claims -> produces output.csv
python code/main.py

# 5. Evaluate against labeled sample data
python code/evaluation/main.py
```

## Approach

### Architecture

```
claims.csv + images + user_history.csv + evidence_requirements.csv
        |
  preprocessor.py   -- load CSVs, resolve image paths, detect prompt injection
        |
  prompt_templates.py -- build structured multi-part prompt (v2)
        |
  vlm_analyzer.py   -- Gemini 2.5 Flash: one API call per claim, all images together
        |
  postprocessor.py  -- deterministic rule engine enforcing consistency
        |
      output.csv
```

### Key Design Decisions

**1. One API call per claim**
All images for a claim are sent in a single Gemini call. This allows the model to reason coherently across multiple evidence images (e.g., multiple angles of the same damage).

**2. Structured JSON output**
`response_mime_type="application/json"` forces Gemini to return valid JSON directly, validated by Pydantic schemas.

**3. Deterministic rule layer (postprocessor.py)**
The VLM output passes through hard rules before writing to output.csv:
- `valid_image=false` → forces `evidence_standard_met=false` AND `claim_status=not_enough_information`
- `wrong_object` flag → forces `claim_status=contradicted`
- User history risk flags are always merged into output risk_flags

**4. Prompt injection detection**
Heuristic detection of embedded instructions in claim text (e.g., "approve this claim"). Flagged as `text_instruction_present` and explicitly warned to the VLM.

**5. Prompt v2 — targeted improvements**
After evaluating on sample_claims.csv (v1 composite: 68.8%), the prompt was improved to fix:
- `glass_shatter` vs `crack` disambiguation (4 cases fixed)
- `wrong_object` image → `contradicted` (not NEI)
- `damage_not_visible` → `contradicted`
- Conservative severity scale (VLM was inflating to `high`)

### Fallback Strategy
When API quota is unavailable, `heuristic_output.py` uses keyword extraction from claim text to predict `issue_type`, `severity`, `object_part`, and `claim_status` without VLM calls. Expected accuracy: ~55-65% vs ~15% for all-NEI baseline.

## File Structure

```
.
├── AGENTS.md                        # Hackathon rules
├── README.md                        # This file
├── output.csv                       # Final predictions on claims.csv
├── code/
│   ├── main.py                      # Entry point: claims.csv -> output.csv
│   ├── config.py                    # Central config (model, delays, paths)
│   ├── schemas.py                   # Pydantic models (VLMOutput, OutputRow)
│   ├── preprocessor.py              # Data loading, injection detection
│   ├── prompt_templates.py          # Prompt builder v2
│   ├── vlm_analyzer.py              # Gemini API calls with retry/resume
│   ├── postprocessor.py             # Rule engine
│   ├── heuristic_output.py          # No-API fallback predictor
│   └── evaluation/
│       └── main.py                  # Evaluation framework with checkpoint/resume
└── dataset/
    ├── sample_claims.csv            # 20 labeled development claims
    ├── claims.csv                   # 44 test claims (no labels)
    ├── user_history.csv             # User risk history
    ├── evidence_requirements.csv    # Minimum image requirements per object
    └── images/
        ├── sample/                  # Images for sample_claims
        └── test/                   # Images for claims.csv
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model override |
| `GEMINI_DELAY` | `90` | Seconds between API calls |
| `GEMINI_MAX_RETRIES` | `8` | Retries per claim on 429 |

## Evaluation Results (v1, sample_claims.csv, n=20)

| Field | Accuracy |
|---|---|
| `evidence_standard_met` | 85% |
| `valid_image` | 85% |
| `risk_flags` (Jaccard) | 74% |
| `claim_status` | 70% |
| `issue_type` | 50% |
| `severity` | 30% |
| **Composite (weighted)** | **68.8%** |

Confusion matrix for `claim_status`:

| GT \ PRED | supported | contradicted | not_enough_info |
|---|---|---|---|
| supported (12) | **12** | 0 | 0 |
| contradicted (5) | 3 | **1** | 1 |
| not_enough_info (3) | 1 | 1 | **1** |

## Rate Limits

Gemini 2.5 Flash free tier: **20 requests/day**. Quota resets at midnight PDT (12:30 PM IST).

The evaluation pipeline supports `--resume` to continue from a checkpoint after a 429 crash:
```bash
python code/evaluation/main.py --resume
python code/evaluation/main.py --fresh   # start over
```
