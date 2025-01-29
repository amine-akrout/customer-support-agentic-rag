from typing import Any, Dict

from llm_guard import scan_prompt
from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity

from src.graph.state import AgentState

input_scanners = [Toxicity(), TokenLimit(), PromptInjection()]


def scan_input_question(state: AgentState) -> Dict[str, Any]:
    """
    Scan the input question.
    """
    question = state["question"]

    sanitized_prompt, results_valid, results_score = scan_prompt(
        input_scanners, question
    )
    if any(not result for result in results_valid.values()):
        state["question_status"] = "invalid"
    else:
        state["question_status"] = "valid"

    state["question"] = sanitized_prompt
    return state
