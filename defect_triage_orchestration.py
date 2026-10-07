import argparse
import io
import logging
import sys
import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

import defect_classifier_agent
import defect_log_analyzer_agent
import defect_recommendation_agent
import defect_similarity_agent
import jira_update_agent
import manual_gate_agent

logger = logging.getLogger(__name__)

VALID_FEATURES = set(defect_classifier_agent.CLASSIFICATION_CATEGORIES)


class DefectTriageState(TypedDict, total=False):
    jira_id: str
    feature_classification: str
    feature: str
    classification: str
    identical_defect_id: str | None
    log_analysis: str
    recommendation_json: dict[str, Any] | list[Any]
    recommendation: str
    approval_status: str
    approval_response: str
    jira_update_status: str
    similar_defects: list[dict[str, Any]]
    defect_title: str
    title: str
    description: str
    timestamp: str | None
    error: str
    warning: str


def _validate_jira_id(state: dict[str, Any]) -> str:
    jira_id = state.get("jira_id")
    if not isinstance(jira_id, str) or not jira_id.strip():
        raise ValueError("state must include a non-empty 'jira_id'.")
    return jira_id.strip()


def _error_update(state: dict[str, Any], stage: str, exc: Exception) -> dict[str, Any]:
    message = f"{stage}: {exc}"
    logger.exception("%s for Jira issue %s", message, state.get("jira_id", "(unknown)"))
    return {"error": message}


def classifier_agent_node(state: DefectTriageState) -> dict[str, Any]:
    """Run classifier against a copy and expose validated graph state updates."""
    try:
        jira_id = _validate_jira_id(state)
        local_state = dict(state)
        local_state["jira_id"] = jira_id
        result = defect_classifier_agent.classifier_node(local_state)
        classification = result.get("feature")
        if not isinstance(classification, str):
            classification = result.get("classification")
        if not isinstance(classification, str):
            raise ValueError("classifier did not return a feature classification.")
        classification = classification.strip().casefold()
        if classification not in VALID_FEATURES:
            raise ValueError(f"classifier returned an unsupported feature: {classification}")

        update: dict[str, Any] = {
            "jira_id": jira_id,
            "feature_classification": classification,
            "feature": classification,
            "classification": classification,
            "defect_title": result.get("defect_title", ""),
            "description": result.get("description", ""),
            "log_analysis": "none",
        }
        for key in ("error", "warning"):
            if key in result:
                update[key] = result[key]
        return update
    except Exception as exc:
        return {
            **_error_update(state, "classifier", exc),
            "feature_classification": "others",
            "feature": "others",
            "classification": "others",
            "log_analysis": "none",
        }


def route_after_classifier(state: DefectTriageState) -> str:
    """Run log analysis only for a recognized, specific feature."""
    feature = state.get("feature_classification", "")
    if isinstance(feature, str) and feature.casefold() in VALID_FEATURES - {"others"}:
        return "log_analyzer"
    logger.info(
        "Skipping log analyzer: feature classification is missing or not specific."
    )
    return "similarity"


def log_analyzer_agent_node(state: DefectTriageState) -> dict[str, Any]:
    """Run the optional log analyzer without mutating the graph's input state."""
    try:
        jira_id = _validate_jira_id(state)
        local_state = dict(state)
        local_state["jira_id"] = jira_id
        local_state["feature"] = state.get("feature_classification", "")
        result = defect_log_analyzer_agent.log_analyzer_node(local_state)
        analysis = result.get("log_analysis", "none")
        if not isinstance(analysis, str):
            raise ValueError("log analyzer must return log_analysis as text.")
        update: dict[str, Any] = {"log_analysis": analysis}
        for key in ("error", "warning"):
            if key in result:
                update[key] = result[key]
        return update
    except Exception as exc:
        return {**_error_update(state, "log analyzer", exc), "log_analysis": "none"}


def similarity_agent_node(state: DefectTriageState) -> dict[str, Any]:
    """Always run similarity search and return only the fields this node owns."""
    try:
        jira_id = _validate_jira_id(state)
        local_state = dict(state)
        local_state["jira_id"] = jira_id
        result = defect_similarity_agent.similarity_node(local_state)
        matches = result.get("similar_defects", [])
        if not isinstance(matches, list):
            raise ValueError("similarity agent must return similar_defects as a list.")
        identical_id = result.get("identical_defect_id")
        if identical_id is not None and not isinstance(identical_id, str):
            raise ValueError("identical_defect_id must be a string or None.")
        update: dict[str, Any] = {
            "similar_defects": matches,
            "identical_defect_id": identical_id,
            "defect_title": result.get("defect_title", state.get("defect_title", "")),
            "description": result.get("description", state.get("description", "")),
            "log_analysis": state.get("log_analysis") or "none",
        }
        for key in ("error", "warning"):
            if key in result:
                update[key] = result[key]
        return update
    except Exception as exc:
        return {
            **_error_update(state, "similarity", exc),
            "similar_defects": [],
            "identical_defect_id": None,
            "log_analysis": state.get("log_analysis") or "none",
        }


def recommendation_agent_node(state: DefectTriageState) -> dict[str, Any]:
    """Generate recommendation JSON from the accumulated graph state."""
    try:
        jira_id = _validate_jira_id(state)
        local_state = dict(state)
        local_state["jira_id"] = jira_id
        local_state["feature"] = state.get("feature_classification", "")
        local_state["log_analysis"] = state.get("log_analysis") or "none"
        result = defect_recommendation_agent.recommendation_node(local_state)
        recommendation_json = result.get("recommendation_json")
        if not isinstance(recommendation_json, (dict, list)):
            raise ValueError("recommendation agent returned no structured JSON.")
        update: dict[str, Any] = {
            "recommendation_json": recommendation_json,
            "recommendation": result.get("recommendation", ""),
        }
        for key in ("error", "warning"):
            if key in result:
                update[key] = result[key]
        return update
    except Exception as exc:
        return _error_update(state, "recommendation", exc)


def manual_gate_agent_node(state: DefectTriageState) -> dict[str, Any]:
    """Use LangGraph's interrupt/resume approval gate."""
    try:
        jira_id = _validate_jira_id(state)
    except Exception as exc:
        return {
            **_error_update(state, "manual approval", exc),
            "approval_status": "invalid id",
        }
    if not state.get("recommendation_json"):
        return {
            "approval_status": "no recommendation",
            "warning": "No recommendation JSON is available for approval.",
        }
    local_state = dict(state)
    local_state["jira_id"] = jira_id
    try:
        # Let LangGraph's GraphInterrupt propagate so it can checkpoint and resume.
        result = manual_gate_agent.manual_gate_node(local_state)
    except GraphInterrupt:
        raise
    except Exception as exc:
        return {
            **_error_update(state, "manual approval", exc),
            "approval_status": "pending",
        }
    return {
        "approval_status": result.get("approval_status", "pending"),
        **{
            key: result[key]
            for key in ("error", "warning")
            if key in result
        },
    }


def route_after_approval(state: DefectTriageState) -> str:
    """Only explicit approval permits the Jira update node to run."""
    return (
        "jira_update"
        if state.get("approval_status") == "approved"
        else "rejected_stop"
        if state.get("approval_status") == "rejected"
        else "approval_stop"
    )


def rejected_stop_node(state: DefectTriageState) -> dict[str, Any]:
    logger.info(
        "Approval was rejected for %s; stopping before Jira update.",
        state.get("jira_id", "(unknown)"),
    )
    return {"jira_update_status": "rejected"}


def approval_stop_node(state: DefectTriageState) -> dict[str, Any]:
    status = state.get("approval_status", "pending")
    logger.warning(
        "Approval status is '%s'; stopping without updating Jira.", status
    )
    return {"jira_update_status": "no update"}


def jira_update_agent_node(state: DefectTriageState) -> dict[str, Any]:
    """Post the recommendation only after explicit approval."""
    if state.get("approval_status") != "approved":
        logger.error("Jira update guard blocked an unapproved recommendation.")
        return {
            "jira_update_status": "no update",
            "error": "Jira update requires explicit human approval.",
        }
    try:
        jira_id = _validate_jira_id(state)
        local_state = dict(state)
        local_state["jira_id"] = jira_id
        result = jira_update_agent.jira_update_node(local_state)
        status = result.get("jira_update_status", "failed")
        if not isinstance(status, str):
            raise ValueError("Jira updater returned an invalid status.")
        update: dict[str, Any] = {"jira_update_status": status}
        for key in ("error", "warning"):
            if key in result:
                update[key] = result[key]
        return update
    except Exception as exc:
        return {**_error_update(state, "Jira update", exc), "jira_update_status": "failed"}


def build_defect_triage_graph(checkpointer: Any | None = None):
    """Construct the complete classifier-to-Jira triage graph."""
    builder = StateGraph(DefectTriageState)
    builder.add_node("classifier", classifier_agent_node)
    builder.add_node("log_analyzer", log_analyzer_agent_node)
    builder.add_node("similarity", similarity_agent_node)
    builder.add_node("recommendation", recommendation_agent_node)
    builder.add_node("manual_gate", manual_gate_agent_node)
    builder.add_node("jira_update", jira_update_agent_node)
    builder.add_node("rejected_stop", rejected_stop_node)
    builder.add_node("approval_stop", approval_stop_node)

    builder.add_edge(START, "classifier")
    builder.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {"log_analyzer": "log_analyzer", "similarity": "similarity"},
    )
    builder.add_edge("log_analyzer", "similarity")
    builder.add_edge("similarity", "recommendation")
    builder.add_edge("recommendation", "manual_gate")
    builder.add_conditional_edges(
        "manual_gate",
        route_after_approval,
        {
            "jira_update": "jira_update",
            "rejected_stop": "rejected_stop",
            "approval_stop": "approval_stop",
        },
    )
    builder.add_edge("jira_update", END)
    builder.add_edge("rejected_stop", END)
    builder.add_edge("approval_stop", END)
    return builder.compile(checkpointer=checkpointer)


def _print_intermediate(state: dict[str, Any], *, streamlit: Any | None = None) -> None:
    if streamlit is None:
        print(f"Classification: {state.get('feature_classification', 'not available')}")
        if state.get("log_analysis") and state["log_analysis"] != "none":
            print(f"Log analysis: {state['log_analysis']}")
        else:
            print("Log analysis: skipped or unavailable")
        print("Similar defects:")
        print(state.get("similar_defects", []))
        print("Recommendation JSON:")
        print(state.get("recommendation_json", {}))
        return

    streamlit.subheader("Classification")
    streamlit.write(state.get("feature_classification", "not available"))
    streamlit.subheader("Log analysis")
    streamlit.write(state.get("log_analysis", "skipped or unavailable"))
    streamlit.subheader("Similar defects")
    streamlit.json(state.get("similar_defects", []))
    streamlit.subheader("Recommendation")
    streamlit.json(state.get("recommendation_json", {}))
    if state.get("error"):
        streamlit.error(state["error"])
    if state.get("warning"):
        streamlit.warning(state["warning"])


def run_cli(jira_id: str, timestamp: str | None = None) -> int:
    """Run the graph sequentially in a resumable CLI session."""
    checkpointer = MemorySaver()
    graph = build_defect_triage_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state: DefectTriageState = {"jira_id": jira_id}
    if timestamp:
        initial_state["timestamp"] = timestamp

    logger.info("Starting defect triage workflow for %s.", jira_id)
    result = graph.invoke(initial_state, config=config)
    if result.get("__interrupt__"):
        _print_intermediate(result)
        while True:
            try:
                answer = input("Approve recommendation and update Jira? [y/n]: ")
            except EOFError:
                print("Approval pending; Jira was not updated.")
                return 1
            normalized = answer.strip().casefold()
            if normalized in {"y", "yes", "approve", "approved"}:
                decision = "approved"
                break
            if normalized in {"n", "no", "reject", "rejected"}:
                decision = "rejected"
                break
            print("Please enter y/yes to approve or n/no to reject.")
        result = graph.invoke(Command(resume=decision), config=config)

    _print_intermediate(result)
    print(f"Approval status: {result.get('approval_status', 'not available')}")
    print(f"Jira update status: {result.get('jira_update_status', 'not updated')}")
    return 1 if result.get("error") else 0


def _invoke_with_streamlit_log_capture(streamlit: Any, invoke: Any) -> Any:
    """Capture all Python logging emitted during a Streamlit workflow step."""
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    try:
        return invoke()
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)
        handler.close()
        previous_log = streamlit.session_state.get("triage_agent_log", "")
        streamlit.session_state["triage_agent_log"] = previous_log + log_buffer.getvalue()


def run_streamlit() -> None:
    """Render the Streamlit UI and resume the graph from explicit button input."""
    import streamlit as st

    st.set_page_config(page_title="Defect Triage", page_icon="🛠️")
    st.title("Defect Triage Workflow")
    jira_id = st.text_input("Jira defect ID", placeholder="FS-74")
    timestamp = st.text_input("Log timestamp (optional)", placeholder="2026-10-06T12:00:00Z")

    if st.button("Run triage", type="primary"):
        if not jira_id.strip():
            st.error("Enter a Jira defect ID.")
            return
        st.session_state["triage_agent_log"] = ""
        thread_id = str(uuid.uuid4())
        checkpointer = MemorySaver()
        graph = build_defect_triage_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        st.session_state["triage_graph"] = graph
        st.session_state["triage_config"] = config
        initial_state: DefectTriageState = {"jira_id": jira_id.strip()}
        if timestamp.strip():
            initial_state["timestamp"] = timestamp.strip()
        with st.spinner("Running defect triage..."):
            result = _invoke_with_streamlit_log_capture(
                st, lambda: graph.invoke(initial_state, config=config)
            )
        st.session_state["triage_result"] = result
        st.session_state["triage_pending"] = bool(result.get("__interrupt__"))

    result = st.session_state.get("triage_result")
    if not isinstance(result, dict):
        return
    _print_intermediate(result, streamlit=st)

    if st.session_state.get("triage_pending"):
        st.info("Review the recommendation and choose Approve or Reject.")
        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve and update Jira", type="primary"):
            with st.spinner("Posting the approved recommendation to Jira..."):
                result = _invoke_with_streamlit_log_capture(
                    st,
                    lambda: st.session_state["triage_graph"].invoke(
                        Command(resume="approved"),
                        config=st.session_state["triage_config"],
                    ),
                )
            st.session_state["triage_result"] = result
            st.session_state["triage_pending"] = False
            st.rerun()
        if reject_col.button("Reject recommendation"):
            result = _invoke_with_streamlit_log_capture(
                st,
                lambda: st.session_state["triage_graph"].invoke(
                    Command(resume="rejected"),
                    config=st.session_state["triage_config"],
                ),
            )
            st.session_state["triage_result"] = result
            st.session_state["triage_pending"] = False
            st.rerun()
    else:
        st.write(f"Approval status: {result.get('approval_status', 'not available')}")
        st.write(f"Jira update status: {result.get('jira_update_status', 'not updated')}")

    st.subheader("Complete agent log")
    with st.container(border=True):
        st.code(
            st.session_state.get("triage_agent_log", "")
            or "No agent logs captured yet.",
            language="text",
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph defect triage workflow."
    )
    parser.add_argument(
        "jira_id",
        nargs="?",
        help="Jira issue key for CLI mode, for example FS-74",
    )
    parser.add_argument(
        "--timestamp",
        help="Optional timestamp to filter Splunk logs in CLI mode.",
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Launch the Streamlit interface instead of the CLI.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logs for workflow execution.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.streamlit or not args.jira_id:
        run_streamlit()
        return 0
    return run_cli(args.jira_id, args.timestamp)


if __name__ == "__main__":
    raise SystemExit(main())
