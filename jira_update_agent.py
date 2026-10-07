import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

load_dotenv(Path(__file__).resolve().with_name(".env"))

logger = logging.getLogger(__name__)

JIRA_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*-\d+$")


def _is_valid_jira_id(jira_id: str) -> bool:
    """Validate Jira issue-key shape locally without making an API request."""
    return bool(JIRA_ID_PATTERN.fullmatch(jira_id.strip()))


def _is_empty_recommendation(recommendation: Any) -> bool:
    if recommendation is None:
        return True
    if isinstance(recommendation, (dict, list)):
        return not recommendation
    if not isinstance(recommendation, str):
        return False
    return recommendation.strip().casefold() in {"", "null", "none"}


def _comment_text(recommendation: Any) -> str:
    """Format structured recommendation data as readable JSON for a Jira comment."""
    if isinstance(recommendation, (dict, list)):
        return json.dumps(recommendation, ensure_ascii=False, indent=2)
    if isinstance(recommendation, str):
        return recommendation.strip()
    raise ValueError("recommendation must be text or a JSON object/array.")


@tool
def add_jira_comment(jira_id: str, recommendation: str) -> str:
    """Add the recommendation as a comment to the specified Jira issue."""
    jira_id = jira_id.strip() if isinstance(jira_id, str) else ""
    recommendation = (
        recommendation.strip() if isinstance(recommendation, str) else ""
    )
    if _is_empty_recommendation(recommendation):
        logger.info("Jira comment creation skipped: recommendation is empty.")
        return "no update"
    if not _is_valid_jira_id(jira_id):
        logger.warning("Jira comment creation skipped: invalid Jira issue key.")
        raise ValueError(f"Invalid Jira issue ID: {jira_id or '(empty)'}")

    base_url = os.getenv("JIRA_BASE_URL", "").rstrip("/")
    jira_email = os.getenv("JIRA_EMAIL")
    jira_api_key = os.getenv("JIRA_API_KEY")
    if not base_url or not jira_email or not jira_api_key:
        raise ValueError(
            "JIRA_BASE_URL, JIRA_EMAIL, and JIRA_API_KEY must be set in .env."
        )

    url = f"{base_url}/rest/api/3/issue/{quote(jira_id, safe='')}/comment"
    payload = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": recommendation}],
                }
            ],
        }
    }
    logger.info("Adding recommendation as a comment to Jira issue %s.", jira_id)
    logger.debug("Sending one Jira comment request to %s.", url)
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Accept": "application/json"},
            auth=(jira_email, jira_api_key),
            timeout=20,
        )
    except requests.RequestException:
        logger.exception("Jira comment request failed for issue %s.", jira_id)
        raise

    logger.debug(
        "Jira comment endpoint responded for %s with HTTP %s.",
        jira_id,
        response.status_code,
    )
    if response.status_code in (400, 404):
        raise LookupError(
            f"Jira rejected issue '{jira_id}' as invalid or inaccessible "
            f"(HTTP {response.status_code})."
        )
    response.raise_for_status()
    logger.info("Recommendation comment added to Jira issue %s.", jira_id)
    return "updated"


def build_jira_update_chain():
    """Build the Jira comment Tool-to-string-output chain."""
    logger.debug("Jira update chain composed: add_jira_comment Tool | StrOutputParser.")
    return add_jira_comment | StrOutputParser()


def _get_recommendation(state: dict[str, Any]) -> Any:
    structured = state.get("recommendation_json")
    if isinstance(structured, (dict, list)) and structured:
        return structured

    recommendation = state.get("recommendation")
    if not _is_empty_recommendation(recommendation):
        return recommendation
    return recommendation


def _read_recommendation_input(value: str, *, force_json: bool = False) -> Any:
    """Read recommendation text or the complete JSON object from a file/string."""
    input_path = Path(value).expanduser()
    is_json_file = force_json or input_path.suffix.casefold() == ".json"
    if is_json_file:
        if not input_path.is_file():
            script_relative_path = Path(__file__).resolve().parent / input_path
            if script_relative_path.is_file():
                input_path = script_relative_path
            else:
                raise ValueError(f"Recommendation JSON file was not found: {value}")

        try:
            with input_path.open("r", encoding="utf-8") as input_file:
                document = json.load(input_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read recommendation JSON '{input_path}': {exc}"
            ) from exc
        source = str(input_path)
    else:
        stripped_value = value.strip()
        if not stripped_value.startswith(("{", "[")):
            return value
        try:
            document = json.loads(stripped_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Recommendation JSON argument is invalid: {exc}") from exc
        source = "command-line JSON"

    if not isinstance(document, (dict, list)):
        raise ValueError("Recommendation JSON must contain an object or array.")
    logger.info(
        "Loaded the complete JSON recommendation from %s (%d top-level entries).",
        source,
        len(document),
    )
    return document


def jira_update_node(state: dict[str, Any]) -> dict[str, Any]:
    """Add the recommendation comment and update state['jira_update_status']."""
    logger.info("Jira update node started.")
    logger.debug("Incoming state keys: %s", sorted(state.keys()))
    state.pop("error", None)
    state.pop("warning", None)

    recommendation = _get_recommendation(state)
    if _is_empty_recommendation(recommendation):
        state["jira_update_status"] = "no update"
        logger.info("Jira API call skipped: recommendation is missing or empty.")
        return state

    jira_id_value = state.get("jira_id")
    jira_id = jira_id_value.strip() if isinstance(jira_id_value, str) else ""
    if not _is_valid_jira_id(jira_id):
        state["jira_update_status"] = "invalid id"
        state["error"] = f"Invalid Jira issue ID: {jira_id or '(empty)'}"
        logger.warning("Jira update skipped because Jira issue ID is invalid.")
        return state

    if not isinstance(recommendation, str):
        if not isinstance(recommendation, (dict, list)):
            state["jira_update_status"] = "failed"
            state["error"] = "recommendation must be text or structured JSON."
            logger.error("Jira update failed: recommendation has an unsupported type.")
            return state
        recommendation = _comment_text(recommendation)
    elif recommendation.lstrip().startswith(("{", "[")):
        try:
            parsed_json = json.loads(recommendation)
        except json.JSONDecodeError:
            parsed_json = None
        if isinstance(parsed_json, (dict, list)):
            recommendation = _comment_text(parsed_json)

    try:
        status = build_jira_update_chain().invoke(
            {"jira_id": jira_id, "recommendation": recommendation}
        )
    except LookupError as exc:
        state["jira_update_status"] = "invalid id"
        state["error"] = str(exc)
        logger.warning("Jira rejected issue %s: %s", jira_id, exc)
        return state
    except Exception as exc:
        state["jira_update_status"] = "failed"
        state["error"] = str(exc)
        logger.exception("Failed to add recommendation comment to %s.", jira_id)
        return state

    state["jira_update_status"] = status.strip() or "updated"
    logger.info(
        "Jira update node completed for %s with status '%s'.",
        jira_id,
        state["jira_update_status"],
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a recommendation comment to a Jira issue."
    )
    parser.add_argument("jira_id", help="Jira issue key, for example FS-74")
    parser.add_argument(
        "recommendation",
        nargs="?",
        help=(
            "Recommendation text, a .json file, or a JSON string; "
            "JSON is posted with all keys and values."
        ),
    )
    parser.add_argument(
        "--recommendation-file",
        help="JSON file whose complete contents should be posted as a Jira comment.",
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
    if args.recommendation is not None and args.recommendation_file:
        parser.error(
            "Provide recommendation text/file positionally or use --recommendation-file, not both."
        )
    if args.recommendation is None and not args.recommendation_file:
        parser.error("Provide recommendation text or a JSON recommendation input.")

    recommendation_input = args.recommendation_file or args.recommendation
    try:
        recommendation = _read_recommendation_input(
            recommendation_input,
            force_json=bool(args.recommendation_file),
        )
    except ValueError as exc:
        parser.error(str(exc))

    logger.info("Starting Jira comment update for %s.", args.jira_id)
    state = jira_update_node(
        {
            "jira_id": args.jira_id,
            "recommendation": recommendation,
        }
    )
    status = state.get("jira_update_status", "failed")
    print(status)
    logger.info("Jira comment update finished with status '%s'.", status)
    return 1 if status in {"failed", "invalid id"} else 0


if __name__ == "__main__":
    raise SystemExit(main())