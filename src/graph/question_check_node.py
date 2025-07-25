"""
LLM Input Safety Check Module

"""

from typing import Any, Dict

import torch
import torch._inductor.config
from llm_guard import scan_prompt
from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity

from src.graph.state import AgentState

torch.set_float32_matmul_precision("high")
torch._inductor.config.fx_graph_cache = True


def scan_prompt_injection(state: AgentState) -> Dict[str, Any]:
    """
    Scan the input question.
    """
    question = state["question"]

    _, results_valid, _ = scan_prompt([PromptInjection(use_onnx=True)], question)
    safe_question = not results_valid.get("PromptInjection", True)
    return {"question_status": [1 if safe_question else 0]}


def scan_toxicity(state: AgentState) -> Dict[str, Any]:
    """
    Scan the input question.
    """
    question = state["question"]
    _, results_valid, _ = scan_prompt([Toxicity(use_onnx=True)], question)
    toxic_question = not results_valid.get("Toxicity", True)
    return {"question_status": [1 if toxic_question else 0]}


def scan_token_limit(state: AgentState) -> Dict[str, Any]:
    """
    Scan the token limit.
    """
    question = state["question"]
    _, results_valid, _ = scan_prompt([TokenLimit(limit=200)], question)
    token_limit_exceeded = not results_valid.get("TokenLimit", True)
    return {"question_status": [1 if token_limit_exceeded else 0]}


def question_check_node(state: AgentState) -> Dict[str, Any]:
    """
    Scan and validate the input question.
    """
    question_status = state["question_status"]
    all_checks_passed = all(status == 0 for status in question_status[-3:])
    if all_checks_passed:
        return {"question": state["question"], "question_valid": True}
    return {
        "llm_output": "Question failed checks, please try again.",
        "question_valid": False,
    }


if __name__ == "__main__":
    state = {"question": "What is the capital of France?"}

    scan_prompt_injection(state)
    scan_toxicity(state)
    scan_token_limit(state)
