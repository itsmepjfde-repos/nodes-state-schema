import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.types import interrupt

load_dotenv(Path(__file__).resolve().with_name(".env"))

logger = logging.getLogger(__name__)

JIRA_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*-\d+$")
APPROVAL_VALUES = {
    "y": "approved",
    "yes": "approved",
    "approve": "approved",
    "approved": "approved",
    "n": "rejected",
    "no": "rejected",
    "reject": "rejected",
    "rejected": "rejected",
}


def _is_valid_jira_id(jira_id: str) -> bool:
    return bool(JIRA_ID_PATTERN.fullmatch(jira_id.strip()))


def _parse_recommendation(value: Any) -> dict[str, Any] | list[Any] | None:
    """Normalize supported recommendation JSON forms."""
    if isinstance(value, (dict, list)):
        return value if value else None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Recommendation input is not valid JSON: {exc}") from exc
    if not isinstance(parsed, (dict, list)):
        raise ValueError("Recommendation JSON must be an object or array.")
    return parsed if parsed else None


def _approval_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return "approved" if value else "rejected"
    if isinstance(value, dict):
        for key in ("approval", "decision", "response"):
            if key in value:
                value = value[key]
                break
        else:
            if "approved" in value and isinstance(value["approved"], bool):
                value = value["approved"]
    if not isinstance(value, str):
        return None
    return APPROVAL_VALUES.get(value.strip().casefold())


@tool
def manual_gate_tool(
    jira_id: str,
    recommendation_json: str,
    approval_response: str = "",
    prompt_for_approval: bool = True,
) -> str:
    """Display recommendation JSON and obtain explicit human approval or rejection."""
    jira_id = jira_id.strip() if isinstance(jira_id, str) else ""
    if not _is_valid_jira_id(jira_id):
        logger.warning("Human approval skipped: Jira issue ID is invalid.")
        return "invalid id"

    try:
        recommendation = _parse_recommendation(recommendation_json)
    except ValueError:
        logger.exception("Human approval skipped: recommendation JSON is invalid.")
        return "no recommendation"
    if recommendation is None:
        logger.info("Human approval skipped: recommendation JSON is missing or empty.")
        return "no recommendation"

    decision = _approval_value(approval_response)
    if decision:
        logger.info("Human decision for %s: %s.", jira_id, decision)
        return decision
    if approval_response and not prompt_for_approval:
        return "invalid"
    if not prompt_for_approval:
        return "pending"

    print(f"\nRecommendation for Jira issue {jira_id}:")
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    while True:
        try:
            answer = input("Approve this recommendation? [y/n]: ")
        except EOFError:
            logger.warning("No approval response received for %s.", jira_id)
            return "pending"
        decision = _approval_value(answer)
        if decision:
            logger.info("Human decision for %s: %s.", jira_id, decision)
            return decision
        print("Please enter y/yes to approve or n/no to reject.")


def build_manual_gate_chain():
    """Build the human-approval Tool-to-string-output chain."""
    logger.debug(
        "Manual gate chain composed: manual_gate_tool Tool | StrOutputParser."
    )
    return manual_gate_tool | StrOutputParser()


def _resume_approval(
    jira_id: str,
    recommendation_json: str,
    initial_response: Any = None,
) -> str:
    response = initial_response
    while True:
        if response is None or not _approval_value(response):
            try:
                response = interrupt(
                    {
                        "type": "human_approval",
                        "jira_id": jira_id,
                        "recommendation_json": json.loads(recommendation_json),
                        "prompt": "Approve this recommendation? Respond with y/yes or n/no.",
                    }
                )
            except RuntimeError:
                logger.info(
                    "LangGraph interrupt is unavailable outside a resumable graph; approval remains pending."
                )
                return "pending"

        response_text = _approval_value(response) or str(response)
        tool_result = build_manual_gate_chain().invoke(
            {
                "jira_id": jira_id,
                "recommendation_json": recommendation_json,
                "approval_response": response_text,
                "prompt_for_approval": False,
            }
        )
        if tool_result != "invalid":
            return tool_result
        logger.warning("Invalid approval response received; asking for a valid decision.")
        response = None


def manual_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Pause triage for approval and update state['approval_status']."""
    logger.info("Manual approval gate started.")
    logger.debug("Incoming state keys: %s", sorted(state.keys()))
    state.pop("error", None)
    state.pop("warning", None)

    jira_id_value = state.get("jira_id")
    jira_id = jira_id_value.strip() if isinstance(jira_id_value, str) else ""
    if not _is_valid_jira_id(jira_id):
        state["approval_status"] = "invalid id"
        state["error"] = f"Invalid Jira issue ID: {jira_id or '(empty)'}"
        logger.warning("Manual approval gate stopped because the Jira ID is invalid.")
        return state

    try:
        recommendation = _parse_recommendation(state.get("recommendation_json"))
    except ValueError as exc:
        state["approval_status"] = "no recommendation"
        state["warning"] = str(exc)
        logger.warning("Manual approval gate stopped: %s", exc)
        return state
    if recommendation is None:
        state["approval_status"] = "no recommendation"
        logger.info("Manual approval gate stopped: recommendation JSON is missing.")
        return state

    recommendation_json = json.dumps(recommendation, ensure_ascii=False, indent=2)
    logger.info("Recommendation is available for approval for %s.", jira_id)
    explicit_response = state.get("approval_response")
    if explicit_response is None:
        explicit_response = state.get("human_approval")

    state["approval_status"] = "pending"
    if state.pop("_interactive_cli", False):
        approval_status = build_manual_gate_chain().invoke(
            {
                "jira_id": jira_id,
                "recommendation_json": recommendation_json,
                "prompt_for_approval": True,
            }
        )
    else:
        approval_status = _resume_approval(
            jira_id,
            recommendation_json,
            initial_response=explicit_response,
        )

    state["approval_status"] = approval_status
    if approval_status == "pending":
        logger.info("Approval is pending; workflow must remain paused.")
    elif approval_status == "approved":
        logger.info("Recommendation approved for %s; workflow may continue.", jira_id)
    elif approval_status == "rejected":
        logger.info("Recommendation rejected for %s; workflow must stop.", jira_id)
    else:
        state["approval_status"] = "no recommendation"
        logger.error("Manual approval tool returned an unexpected status.")
    return state


def route_after_approval(state: dict[str, Any]) -> str:
    """Return the LangGraph route; only explicit approval continues to Jira update."""
    if state.get("approval_status") == "approved":
        return "continue"
    return "stop"


def _load_recommendation_input(value: str, force_file: bool = False) -> Any:
    input_path = Path(value).expanduser()
    is_file = force_file or input_path.suffix.casefold() == ".json"
    if is_file:
        if not input_path.is_file():
            script_relative_path = Path(__file__).resolve().parent / input_path
            if script_relative_path.is_file():
                input_path = script_relative_path
            else:
                raise ValueError(f"Recommendation JSON file was not found: {value}")
        try:
            with input_path.open("r", encoding="utf-8") as input_file:
                content = json.load(input_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read recommendation JSON: {exc}") from exc
    else:
        content = value
    recommendation = _parse_recommendation(content)
    if recommendation is None:
        raise ValueError("Recommendation JSON is empty.")
    return recommendation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Request human approval for a Jira defect recommendation."
    )
    parser.add_argument("jira_id", help="Jira issue key, for example FS-74")
    parser.add_argument(
        "recommendation",
        help="Recommendation JSON string or path to a .json file.",
    )
    parser.add_argument(
        "--recommendation-file",
        action="store_true",
        help="Treat the recommendation argument as a JSON file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level execution logs.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        recommendation_json = _load_recommendation_input(
            args.recommendation,
            force_file=args.recommendation_file,
        )
    except ValueError as exc:
        parser.error(str(exc))

    state = manual_gate_node(
        {
            "jira_id": args.jira_id,
            "recommendation_json": recommendation_json,
            "_interactive_cli": True,
        }
    )
    status = state["approval_status"]
    print(status)
    return 0 if status in {"approved", "rejected", "no recommendation"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
