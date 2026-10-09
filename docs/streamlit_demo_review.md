# Streamlit competition presentation review

**Historical review, before UI polish.** The profile wiring and presentation
issues below were subsequently addressed. See [ui_polish_status.md](ui_polish_status.md)
for the current IncidentLens interface, screenshots and actual runtime checks.

Reviewed on 2026-10-09 in real headless Chrome through Playwright at
1920x1080 and 1366x768. This review used the actual Streamlit server and actual
button/selectbox interactions, not only AppTest or code inspection.

**Verdict: suitable for a labeled mock technical walkthrough, but not yet ready
for a polished competition presentation with real RTX inference.** No application
code was modified in this review.

## What worked

- Load Demo Incident and Analyze Incident completed the mock pipeline in-browser.
- Primary diagnosis and heuristic score were correct, with all three ranked
  hypotheses, missing observations, supporting evidence and recommended action.
- Mock execution and conservative causality were explicitly labeled.
- Input files, evidence, relations, verification, routing, telemetry and artifact
  links/download controls were rendered. Browser console reported zero errors
  and zero warnings during the reviewed session.

## Required before a real-GPU competition demo

### P0: Streamlit CUDA profile is not wired to the validated RTX profile

Selecting Local Transformers (real) and cuda, then Analyze Incident, failed:

```text
Phase 3 run failed (ValueError): No eligible model profile:
{'local-small': 'Configured memory estimate exceeds available memory in the same memory scope.'}
```

`app/ui.py:55` builds `default_profiles`, whose local estimate is 8192 MiB
(`app/phase3.py:37`). Measured available CUDA memory was 7033 MiB. The separately
validated `profiles/demo_rtx5060.json` uses a conservative 4096 MiB estimate,
but this UI does not select it. CUDA itself works through the existing CLI;
this failure is profile integration, not evidence of a broken CUDA driver/model.
No CPU or mock fallback occurred. The failure artifact is:

`runs/20261009T083025.950897Z-48b45cb2/`

Connect the UI's real CUDA option to the validated development profile and
retain the existing memory checks and explicit failure behavior. Label RTX as
development hardware and Intel Arc Pro B70 as the unvalidated competition target.

### P1: move the useful result into the first screen

At 1920x1080, Primary diagnosis starts around y=922; its score is cut off at the
bottom. The result page has a 11650-pixel main scroll area. At 1366x768 the full
main content is about 12434 pixels tall and the result is below the first screen.
Input/upload controls and four file expanders consume the first screen even
after diagnosis finishes. A presenter must scroll repeatedly to tell the story.

Use a compact result summary: root cause, evidence score caveat, actual inference
mode/model/device, latency, three ranked alternatives and recommended action.
Make the 38 GiB projected demand versus nominal 32 GiB capacity visible with its
derived-evidence references. Keep full inputs and provenance available below.

### P1: present summaries before raw diagnostic detail

Ranked hypotheses has a useful three-row table, followed by fully expanded JSON
for every hypothesis. Routing and telemetry also default to expanded JSON.
These are useful audit details but obscure the product demonstration. Show a
short verification-status table, routing explanation and metric cards first;
place raw JSON in collapsed detail sections without removing information.

Supporting evidence occupies a long left column; the right contradicting column
is largely empty for this scenario. The first supporting entries are service
failure/startup events, while the distinguishing memory/config/code facts appear
later. A brief, evidence-cited summary of those key observations should lead;
retain all excerpts in expandable details. Long source paths and 14px body text
also make projected screenshots harder to scan than a concise summary would.

## Screenshots

The screenshot files are under `output/playwright/`:

- `competition-inputs-1920.png`: loaded incident inputs.
- `competition-result-top-1920.png`: completed mock result at the first screen.
- `competition-primary-1920.png`: diagnosis and supporting-evidence layout.
- `competition-ranking-1920.png`: ranking table and expanded verification JSON.
- `competition-result-top-1366.png`: completed mock result at the smaller viewport.
- `competition-cuda-blocker-1920.png`: actual CUDA UI failure.
- `competition-cuda-blocker-1366.png`: same failure at the smaller viewport.

The previous full suite passed 161 tests after CUDA environment validation.
This review did not change application code or rerun that suite; real-browser
interaction and screenshots add presentation evidence that AppTest alone cannot
provide. No B70 run, physical projector test or real CUDA success through the UI
is claimed. A successful CPU real-model AppTest from the earlier suite does not
cover the CUDA UI configuration gap found here.

After the three fixes above, repeat the real-browser CUDA workflow, review both
screen sizes and rehearse the complete three-minute demonstration.
