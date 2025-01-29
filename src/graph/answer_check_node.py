from typing import Any, Dict

from llm_guard import scan_output
from llm_guard.output_scanners import LanguageSame, Relevance, Sentiment

from src.graph.state import AgentState

output_scanners = [
    LanguageSame(),
    Relevance(),
    Sentiment(),
]


def scan_output_answer(state: AgentState) -> Dict[str, Any]:
    """
    Scan the output answer.
    """
    output = state["llm_output"]
    prompt = state["prompt"]
    sanitized_response, results_valid, results_score = scan_output(
        scanners=output_scanners, output=output, prompt=prompt
    )

    if any(not result for result in results_valid.values()):
        state["answer_status"] = "invalid"
    else:
        state["answer_status"] = "valid"
        state["llm_output"] = sanitized_response

    return state
