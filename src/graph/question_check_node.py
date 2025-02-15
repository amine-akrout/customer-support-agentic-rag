from typing import Any, Dict

from llm_guard import scan_prompt

from src.graph.state import AgentState


def scan_input_question(state: AgentState, input_scanners) -> Dict[str, Any]:
    """
    Scan the input question.
    """
    question = state["question"]

    sanitized_prompt, results_valid, _ = scan_prompt(input_scanners, question)
    if any(not result for result in results_valid.values()):
        state["question_status"] = "invalid"
        state["llm_output"] = "Please provide a valid question with clear intent."
    else:
        state["question_status"] = "valid"

    state["question"] = sanitized_prompt
    return state


if "__name__" == "__main__":
    from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity

    input_scanners = [
        PromptInjection(),
        TokenLimit(),
        Toxicity(),
    ]
    state = {"question": "What is the capital of France?"}
    scan_input_question(state, input_scanners)
    print(state)
