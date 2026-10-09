"""IncidentLens presentation; diagnosis and artifact contracts unchanged."""
import json
import os
from pathlib import Path
import streamlit as st
from app.main import DEMO_DIRECTORY
from app.phase3 import Phase3RunError, run_phase3
from app.ui_presenter import (display_path, format_value, incident_overview, presentation,
                              runtime_summary, supporting_summary, verification_status)
from app.ui_runtime import detect_cuda, ui_profiles
from core.correlation import correlate
from core.evidence_extractor import extract_evidence
from core.evidence_repository import EvidenceRepository
from core.incident_loader import IncidentFile, load_incident


@st.cache_data(ttl=30, max_entries=1)
def available_cuda():
    return detect_cuda()


@st.cache_data(max_entries=4)
def overview_for(files):
    repository = EvidenceRepository(extract_evidence(files))
    derived, _ = correlate(repository)
    repository.extend(derived)
    return incident_overview([e.model_dump(mode="json") for e in repository.all()])


def evidence_bullets(items, capacity=None):
    for item in items:
        value = format_value(item["value"], item["unit"]) if item["value"] is not None else ""
        if item["id"].endswith(":service.workers"):
            text = f"{value} inference workers configured"
        elif item["id"].endswith(":model.load_per_worker"):
            text = f"Worker-local model loading configured: {value}"
        elif item["id"].endswith("derived:projected_memory_gib"):
            text = f"Projected demand: {value}"
        elif item["id"].endswith(":runtime.gpu_memory_limit_gb"):
            text = f"Configured GPU capacity: {capacity['value'] if capacity and capacity['evidence_id'] == item['id'] else value}"
        elif item["id"].endswith(":oom"):
            text = item["statement"]
        elif item["category"] == "code_behavior":
            text = item["statement"]
        else:
            text = item["statement"] + (f" · {value}" if value else "")
        st.markdown(f"- {text}")
        st.caption(item["id"])


st.set_page_config(page_title="IncidentLens", page_icon=":material/troubleshoot:", layout="wide")
st.title("IncidentLens")
st.markdown("**Evidence-Grounded Local Incident Diagnosis**")
st.caption("Turn fragmented logs, code, configurations, and metrics into verifiable root-cause hypotheses — entirely on local hardware.")
cuda = available_cuda()
st.session_state.setdefault("mode", "Local Transformers (real)")
st.session_state.setdefault("device", "cuda" if cuda["available"] else "cpu")
with st.sidebar:
    st.subheader("Runtime")
    with st.expander("Advanced Runtime Settings", expanded=False):
        mode = st.selectbox("Inference mode", ["Mock (deterministic)", "Local Transformers (real)"], key="mode")
        model_path = st.text_input("Existing local model directory", value=os.environ.get("INCIDENT_MODEL_PATH", "models/Qwen2.5-0.5B-Instruct"), key="model_path")
        device = st.selectbox("Device", ["cpu", "cuda", "xpu"], key="device")
        preference = st.selectbox("Latency preference", ["balanced", "fast", "quality"], key="preference")
        st.caption("Weights must already exist locally. No silent inference fallback.")
    st.markdown(f"**Inference:** {'Local AI' if mode.startswith('Local') else 'Mock / development'}")
    st.markdown(f"**Model:** {Path(model_path).name if mode.startswith('Local') else 'Deterministic mock'}")
    actual_device = cuda["name"] if device == "cuda" and cuda["available"] else device.upper()
    st.markdown(f"**Device:** {actual_device if mode.startswith('Local') else 'No model executed'}")
    st.markdown(f"**Optimization:** {preference.title()}")
    if not cuda["available"]:
        st.warning(cuda["warning"])
    if device == "cuda" and not cuda["available"]:
        st.warning("CUDA was explicitly selected but is unavailable. This run will fail rather than use CPU silently.")

with st.container(horizontal=True):
    if st.button("Run GPU OOM Demo", type="primary", icon=":material/play_arrow:", key="load_demo"):
        st.session_state["inputs"] = load_incident(DEMO_DIRECTORY)
        st.session_state["is_demo"] = True
        st.session_state.pop("run", None)
    if st.button("Upload Your Incident", icon=":material/upload:", key="upload_incident"):
        st.session_state["show_upload"] = True
if st.session_state.get("show_upload"):
    uploaded = st.file_uploader("Incident source files", type=["log", "yaml", "yml", "py", "txt"], accept_multiple_files=True)
    if uploaded:
        try:
            files = [IncidentFile(file_path=Path(f.name).name, content=f.getvalue().decode("utf-8")) for f in uploaded]
            if files != st.session_state.get("inputs"):
                st.session_state["inputs"] = files
                st.session_state["is_demo"] = False
                st.session_state.pop("run", None)
        except UnicodeDecodeError:
            st.error("Incident sources must be UTF-8 text.")
            st.stop()
files = st.session_state.get("inputs", [])
run = st.session_state.get("run")
if not files:
    capabilities = [
        ("Evidence Grounded", "Every diagnosis is linked to traceable evidence."),
        ("Verification First", "Hypotheses are tested before ranking."),
        ("Private Local AI", "Engineering data stays on local hardware."),
    ]
    for column, (title, description) in zip(st.columns(3, border=True), capabilities):
        with column:
            st.markdown(f"**{title}**")
            st.caption(description)
if files:
    try:
        overview = overview_for(files)
        with st.expander("Incident Overview", expanded=run is None):
            if st.session_state.get("is_demo"):
                st.markdown("**Local AI inference service** · Synthetic worker startup / GPU allocation failure")
            for column, (label, item) in zip(st.columns(4), overview.items()):
                column.metric(label, item["value"], help=f"Evidence: {item['evidence_id'] or 'unavailable'}")
            st.caption("Sources: " + " · ".join(Path(f.file_path).name for f in files))
        if st.session_state.get("is_demo"):
            st.caption("Synthetic incident capacity, separate from inference hardware. "
                       f"Projected demand: {overview['Projected Demand']['value']} · Capacity: {overview['GPU Capacity']['value']}")
    except ValueError as exc:
        st.warning(f"Overview unavailable: {exc}. Analyze Incident to record the full pipeline outcome.")

if st.button("Analyze Incident", type="primary", disabled=not files, key="analyze"):
    st.session_state.pop("run", None)
    run = None
    backend = "mock" if mode.startswith("Mock") else "transformers"
    try:
        with st.status("Analyzing local incident", expanded=True) as status:
            profiles = ui_profiles(backend, model_path if backend == "transformers" else None, device)
            run = run_phase3(files, profiles=profiles, execution_mode="mock" if backend == "mock" else "local",
                             device=device, latency_preference=preference,
                             save_root=os.environ.get("INCIDENT_RUNS_DIR", "runs"), progress=st.write)
            st.session_state["run"] = run
            status.update(label="Diagnosis complete; artifacts saved", state="complete", expanded=False)
        st.rerun()
    except (Phase3RunError, ValueError, OSError) as exc:
        message = str(exc)
        if "; artifacts: " in message:
            detail, artifact_path = message.rsplit("; artifacts: ", 1)
            message = f"{detail}; artifacts: {display_path(artifact_path)}"
        st.error(message)

if run:
    view = presentation(run)
    runtime = runtime_summary(run)
    st.info("MOCK inference — no language model executed." if view["mode"] == "mock"
            else f"REAL local inference · {runtime['model']} · {runtime['device']}")
    with st.container(border=True):
        st.subheader("Diagnosis")
        title, score = st.columns([3, 1])
        with title:
            st.caption("Highest-Ranked Hypothesis")
            st.markdown(f"### {view['title']}")
        score.metric("Evidence Score", f"{view['confidence'] * 100:.1f}%")
        st.caption("Independent heuristic score, not a probability. " + view["conclusion"])
    with st.container(border=True):
        support, contra, missing = st.columns([2, 1, 1])
        with support:
            st.subheader("Supporting Evidence")
            evidence_bullets(supporting_summary(run), overview["GPU Capacity"])
        with contra:
            st.subheader("Contradicting Evidence")
            if view["contradicting"]:
                evidence_bullets(view["contradicting"])
            else:
                st.caption("No strong contradicting evidence identified in the supplied sources.")
        with missing:
            st.subheader("Missing Evidence")
            for observation in view["missing"]:
                st.markdown(f"- {observation}")
            if not view["missing"]:
                st.caption("No missing criteria reported; causality remains unproven.")
    st.subheader("Alternative Hypotheses")
    st.caption("Includes the leader for comparison. Independent scores; not normalized to sum to 100%.")
    for hypothesis in view["hypotheses"]:
        st.progress(hypothesis["confidence"], text=f"{hypothesis['title']} — {hypothesis['confidence'] * 100:.1f}%")
    st.subheader("Verification")
    primary = run["diagnosis"]["primary_root_cause"]
    results = primary["verification_results"] if primary else []
    st.markdown(f"**{verification_status(results)}** · {view['title']}")
    st.markdown("**The system does not claim causality until sufficient evidence is available.**")
    counts = {key: sum(r["status"] == key for r in results) for key in ["supported", "contradicted", "mixed", "missing"]}
    for column, (key, count) in zip(st.columns(4), counts.items()):
        column.metric(key.title(), count, border=True)
    st.markdown(view["verification"])
    st.subheader("Recommended Next Step")
    st.markdown(view["action"])
    st.caption("Recommendation only. No experiment or remediation is executed.")
    st.subheader("Local AI Runtime")
    execution_device = "No model executed (mock)" if view["mode"] == "mock" else runtime["device"]
    st.markdown(f"**Model:** {runtime['model']} · **Actual device:** {execution_device}")
    metrics = st.columns(4)
    metrics[0].metric("End-to-End Latency", f"{runtime['end_to_end_s']:.2f} s")
    metrics[1].metric("Generation Latency", f"{runtime['generation_s']:.2f} s" if runtime["generation_s"] is not None else "N/A")
    metrics[2].metric("Throughput", f"{runtime['tokens_per_second']:.2f} tokens/s" if runtime["tokens_per_second"] is not None else "N/A")
    metrics[3].metric("GPU Memory Snapshot", f"{runtime['gpu_snapshot_mb']:.1f} MiB" if runtime["gpu_snapshot_mb"] is not None else "N/A",
                      help="Largest post-call PyTorch allocated-memory snapshot in this run; not peak or whole-device VRAM.")
    st.caption(f"Routing overhead: {runtime['routing_ms']:.4f} ms. Latency excludes artifact writing and rendering.")
    shown_profiles = set()
    for decision in view["routing"]:
        if decision["selected_profile"]["id"] in shown_profiles:
            continue
        shown_profiles.add(decision["selected_profile"]["id"])
        st.markdown(f"**Routing Decision:** {decision['selected_profile']['id']}")
        st.caption(decision["reason"])
    st.caption("Development validation hardware: NVIDIA GeForce RTX 5060 Laptop GPU. "
               "Competition target platform: Intel Arc Pro B70 32 GB. B70 validation: pending / not yet validated. "
               "RTX measurements are not B70 performance estimates.")
    with st.expander("View Extracted Evidence"):
        st.dataframe(view["evidence"], hide_index=True, alt="Extracted observations and derived facts with evidence IDs")
    with st.expander("View Evidence Relations"):
        st.dataframe(view["relations"], hide_index=True, alt="Evidence relation endpoints and explanations")
    with st.expander("View Raw Verification Details"):
        st.dataframe(view["verification_results"], hide_index=True, alt="All hypothesis verification results")
        st.dataframe(view["hypotheses"], hide_index=True, alt="Exact hypothesis scores and coverage")
        st.json(run["diagnosis"]["hypotheses"], expanded=False)
    with st.expander("View Run Metadata"):
        st.caption("Raw audit metadata may contain original local paths. Saved provenance and the JSON download are unchanged.")
        st.json(run["metadata"], expanded=False)
        st.json(view["routing"], expanded=False)
        st.json(view["telemetry"], expanded=False)
        st.caption(f"Run artifacts: {display_path(run['artifact_directory'])}")
    st.download_button("Download result JSON", json.dumps(run, indent=2), file_name=f"{run['run_id']}.json", mime="application/json")
if files:
    with st.expander("View Incident Inputs"):
        for file in files:
            st.markdown(f"**{Path(file.file_path).name}**")
            st.code(file.content, language="python" if file.file_path.endswith(".py") else "text")
