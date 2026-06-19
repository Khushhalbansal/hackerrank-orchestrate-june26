# Damage Claim Verification System

A multi-modal evidence review pipeline that verifies damage claims using Gemini 2.5 Flash (VLM), structured prompting, and a deterministic rule layer.

## Setup

```bash
pip install google-genai pandas pillow tqdm pydantic
set GEMINI_API_KEY=your_key_here   # Windows
# export GEMINI_API_KEY=...         # macOS/Linux
```

## File Structure

```
code/
├── main.py                  # Entry point for test claims
├── config.py                # Paths, model settings, rate limits
├── schemas.py               # Pydantic models (enums + VLMOutput + OutputRow)
├── preprocessor.py          # CSV loading, image resolution, injection detection
├── prompt_templates.py      # Prompt builder with anti-injection rules
├── vlm_analyzer.py          # Gemini API call + retry logic
├── postprocessor.py         # Rule engine: flag merging, consistency checks
└── evaluation/
    ├── main.py              # Evaluation on sample_claims.csv
    └── evaluation_report.md # Auto-generated after running evaluation
```

## Run

### Test all claims (produces output.csv):
```bash
python code/main.py
```

### Run on sample claims only (for debugging):
```bash
python code/main.py --input dataset/sample_claims.csv --output output_sample.csv
```

### Evaluate against labeled sample data:
```bash
python code/evaluation/main.py
```

### Adjust rate-limit delay:
```bash
python code/main.py --delay 6    # 6 seconds between calls (safe for free tier)
```

## Architecture

```
claims.csv + images + user_history + evidence_requirements
         |
   preprocessor.py   (load, resolve paths, detect injection)
         |
   prompt_templates.py  (build structured prompt)
         |
   vlm_analyzer.py   (Gemini 2.5 Flash: 1 call per claim, all images)
         |
   postprocessor.py  (merge history flags, enforce consistency rules)
         |
       output.csv
```

## Key Design Decisions

- **One API call per claim** — all images sent together for coherent reasoning
- **Anti-injection** — prompt explicitly warns Gemini about embedded instructions
- **Deterministic rule layer** — `valid_image=false` always cascades to `evidence_standard_met=false` and `claim_status=not_enough_information`
- **Rate limit safe** — 6s delay → stays within 10 RPM free-tier limit
- **Pydantic validation** — VLM output is schema-validated before writing to CSV
