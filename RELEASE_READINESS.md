# Submission readiness audit

Audited 2026-10-10 (Asia/Taipei). Final Git submission audit: **PASS — READY TO
STAGE FOR FIRST COMMIT**. The project Git repository was initialized with user
authorization during this follow-up audit. Its index is empty; no staging,
commit or publishing was performed. Application validation below is retained
from the preceding release audit; no application code changed in this follow-up.

| Item | Status | Evidence / limitation |
| --- | --- | --- |
| 1. Fresh README installation | PASS | Created an isolated Python environment and installed editable `.[dev,ui]`, then the documented pinned CUDA runtime. Existing package download cache and pinned local model weights were reused; this was not a clean Git clone or a repeated model download. |
| 2. Fresh Streamlit launch | PASS | Clean-environment server on port 8502; actual browser verified initial screen, controls, diagnosis and runtime panels. |
| 3. Real GPU OOM demo | PASS | Qwen2.5-0.5B-Instruct made three local CUDA inference calls on RTX 5060 Laptop; normal artifact saved and checksums verified. |
| 4. No silent fallback | PASS | Successful run used CUDA for every inference call. A deliberately invalid real/CUDA configuration showed an explicit routing error, saved a failed artifact and removed the prior diagnosis; no mock/CPU result appeared. Existing backend failure tests also passed. |
| 5. Metrics belong to current run | PASS | All four displayed runtime metrics matched the successful run artifact. Presentation tests verify distinct runs retain their own values. Memory is an allocated snapshot, not peak VRAM. |
| 6. Actual Git tracked files | PASS | Real project Git repository verified. Index empty; all 69 eligible candidate files inspected. No weights, caches, secrets, generated output or large binaries in the proposed inventory. There are no commits/history to audit yet. |
| 7. `.gitignore` | PASS | Rechecked against actual Git: models, weights, caches, runs, browser output, local secrets and keys excluded. Added exact exclusions for two virtual-environment skill symlinks and separately submitted competition consent paperwork. Required example logs and Streamlit theme configuration remain eligible. |
| 8. Presentation paths | WARN | Evidence sources, artifact captions and failure artifact paths now display relative paths or basenames. Raw audit JSON/downloads and editable runtime configuration intentionally preserve provenance/configuration; review these before public sharing. |
| 9. Hardware distinctions | PASS | README distinguishes validated RTX 5060 development CUDA execution from unvalidated Intel Arc Pro B70 / XPU target execution. No B70 benchmark numbers are claimed. |
| 10. Benchmark consistency | WARN | README records the fresh result below and scopes previous 37.16 s / 32.40 s dependency-warm results. The 8.85 s / 4.68 s result and older CPU timings are explicitly historical. Cold-start latency remains variable. |
| 11. Existing tests | PASS | Full clean-environment suite: **181 passed in 401.05 s**, including optional local-model tests with existing offline weights. |
| 12. Readiness report | PASS | This report records passes, warnings and the handoff blocker. |

## Current live measurement

Successful artifact: `runs/20261009T155419.822868Z-03e0695d/`.
Failed configuration artifact: `runs/20261009T160112.138581Z-fcca9c44/`.

| Metric | Fresh Streamlit CUDA run |
| --- | --- |
| End-to-end diagnosis | 152.5616 s |
| Model generation | 33.8143 s |
| Input / output tokens | 5,416 / 188 |
| Generation throughput | 5.5598 tokens/s |
| GPU Memory Snapshot | 1,917.3 MiB allocated; peak unavailable |
| Routing overhead | 0.3211 ms |

Validation environment: Windows, Python 3.13.13, PyTorch 2.14.1+cu130,
CUDA 13.0, Transformers 4.57.6, Streamlit 1.65.0; NVIDIA GeForce RTX 5060
Laptop GPU. Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
This first run in a fresh process includes lazy dependency/model startup.
The 118.75 s outside generation is measured aggregate overhead, not a separately
profiled attribution to any particular startup operation. No timing was hidden,
normalized, or replaced with the historical faster result.

The complete ranked hypotheses, scores and verification results match the prior
GPU benchmark: duplicate loading 0.866667, runtime allocation 0.153333, memory
leak 0.113333. Core diagnosis, inference, scoring, verification, routing and
artifact code remained unchanged. Presentation sanitization leaves original
artifact provenance intact.

## Changes and submission follow-up

Audit fixes are limited to presentation path formatting and tests, ignore rules,
the tested Streamlit dependency floor (1.65), and installation/performance docs.
Screenshots are under `output/playwright/release-audit-*.png`; these and run
artifacts are deliberately ignored and will not accompany a future Git clone
unless selected submission attachments are separately curated.

**Blocking issues:** none for staging the first commit. Public repository
publishing and handoff have not been performed or verified; those are subsequent
submission steps, outside this Git-only audit.

Remaining warnings: rehearse cold-start timing on the presentation machine;
validate B70 separately; review raw provenance before sharing artifacts; curate
the demo recording and screenshots. These are not claims of completed target
hardware or submission-material validation.

## Final Git submission audit

Scope: actual empty project index plus every untracked file eligible under
`.gitignore`. Inventory obtained from NUL-delimited `git ls-files --others
--exclude-standard`; every candidate's full UTF-8 content was scanned. The
administrative DOCX was also inspected as XML before being excluded. No file
larger than 5 MiB was eligible. Source, tests, six synthetic incident fixture
files, README, documentation, profiles, `pyproject.toml` and Streamlit theme
configuration are all eligible. Only `.gitignore` and this report changed;
application functionality and schemas were not modified. Tests were not rerun
for these Git/documentation-only changes; the preceding 181-test result stands.

| Check | Status | Result |
| --- | --- | --- |
| Git repository / index | PASS | Real repository initialized; zero tracked/staged files and no commits. |
| Candidate content / secrets | PASS | No API keys, tokens, assigned passwords/credentials, email addresses or actual private identifiers found by pattern scanning and contextual review. This is not a guarantee against every possible undisclosed secret. |
| Absolute local paths | PASS | Only synthetic test paths in `tests/test_ui_presentation.py:73-74`: Windows user `presenter` and POSIX `/home/presenter`. They test redaction and are retained; no actual developer workspace path found. |
| Sizes / generated files | PASS | Largest candidate is the Markdown specification, 29,131 bytes. No unexpected binaries or generated run/test/benchmark output eligible. |
| Required source eligibility | PASS | All required source, tests, synthetic logs/config/expected results, docs and project/UI configuration included below. |
| Ignore rules | PASS | Actual `git check-ignore` and ignored-status listing confirm local model/cache/output exclusions. No existing tracked files bypass ignore rules, because index is empty. |
| Public material limitations | WARN | Run artifacts/screenshots remain local and ignored. B70 validation, cold-start timing rehearsal, material curation and future public handoff remain separate tasks. |

Ignored categories: `.venv/`, `.uv-cache/`, Python bytecode and pytest caches,
egg-info/build/dist, `models/`, `.cache/`, `.hf-cache/`, `runs/`, `output/`,
`.playwright-cli/`, `.env` and private variants, Streamlit secrets, local Claude
settings, model weight formats and private key formats. `.env.example` and
`.env.sample` are permitted intentionally (neither exists in this inventory).
The two exact Streamlit skill symlinks under `.agents/skills/` and
`.claude/skills/` point into the ignored virtual environment and are excluded.
The competition consent DOCX is excluded as administrative paperwork, not
application documentation; it remains available locally for separate submission.
No blanket log exclusion was added: the synthetic `app.log` / `system.log`
remain public demo inputs. Future files still require review before staging.

Largest candidate files (bytes):

| File | Bytes |
| --- | ---: |
| `CODEX_SPEC_Incident_Diagnosis_Agent_v0.1.md` | 29,131 |
| `README.md` | 13,940 |
| `docs/final_demo_polish.md` | 13,406 |
| `app/ui.py` | 13,213 |
| `docs/architecture.md` | 12,757 |

### Exact proposed tracked-file inventory

69 files, including this report. These are proposed files, not staged files.

```text
.gitignore
.streamlit/config.toml
CODEX_SPEC_Incident_Diagnosis_Agent_v0.1.md
README.md
RELEASE_READINESS.md
app/__init__.py
app/main.py
app/phase3.py
app/ui.py
app/ui_presenter.py
app/ui_runtime.py
core/__init__.py
core/confidence.py
core/correlation.py
core/diagnosis.py
core/evidence_extractor.py
core/evidence_repository.py
core/hypothesis.py
core/incident_loader.py
core/router.py
core/run_artifacts.py
core/schemas.py
core/verifier.py
docs/architecture.md
docs/demo_scenario.md
docs/final_demo_polish.md
docs/limitations.md
docs/phase1_status.md
docs/phase2_status.md
docs/phase3_status.md
docs/rtx5060_validation.md
docs/streamlit_demo_review.md
docs/ui_polish_status.md
examples/gpu_oom_duplicate_models/app.log
examples/gpu_oom_duplicate_models/config.yaml
examples/gpu_oom_duplicate_models/expected_diagnosis.json
examples/gpu_oom_duplicate_models/expected_evidence.json
examples/gpu_oom_duplicate_models/inference.py
examples/gpu_oom_duplicate_models/system.log
inference/__init__.py
inference/base.py
inference/mock_backend.py
inference/routed_backend.py
inference/telemetry.py
inference/transformers_backend.py
profiles/demo_local.json
profiles/demo_rtx5060.json
pyproject.toml
tests/conftest.py
tests/test_compact_candidates.py
tests/test_confidence.py
tests/test_correlation.py
tests/test_demo_core.py
tests/test_diagnosis.py
tests/test_evidence_extractor.py
tests/test_evidence_repository.py
tests/test_hypothesis.py
tests/test_inference.py
tests/test_loader.py
tests/test_phase2_confidence.py
tests/test_phase3_pipeline.py
tests/test_real_model.py
tests/test_router.py
tests/test_run_artifacts.py
tests/test_schemas.py
tests/test_telemetry.py
tests/test_ui.py
tests/test_ui_presentation.py
tests/test_verifier.py
```
