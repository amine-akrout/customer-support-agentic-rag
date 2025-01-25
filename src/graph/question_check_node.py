from typing import Any, Dict

from llm_guard import scan_prompt
from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity

from src.graph.state import GraphState

input_scanners = [Toxicity(), TokenLimit(), PromptInjection()]


def scan_input_question(state: GraphState) -> Dict[str, Any]:
    """
    Scan the input question.
    """
    question = state["question"]

    sanitized_prompt, results_valid, results_score = scan_prompt(
        input_scanners, question
    )
    if any(not result for result in results_valid.values()):
        return {"question_status": "invalid"}
    return {"question_status": "valid", "question": sanitized_prompt}
