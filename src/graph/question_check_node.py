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
    if results_valid.get("PromptInjection") == False:
        return {"question_status": [1]}
    else:
        return {"question_status": [0]}


def scan_toxicity(state: AgentState) -> Dict[str, Any]:
    """
    Scan the input question.
    """
    question = state["question"]
    _, results_valid, _ = scan_prompt([Toxicity(use_onnx=True)], question)
    if results_valid.get("Toxicity") == False:
        return {"question_status": [1]}
    else:
        return {"question_status": [0]}


def scan_token_limit(state: AgentState) -> Dict[str, Any]:
    """
    Scan the token limit.
    """
    question = state["question"]
    _, results_valid, _ = scan_prompt([TokenLimit(limit=200)], question)
    if results_valid.get("TokenLimit") == False:
        return {"question_status": [1]}
    else:
        return {"question_status": [0]}


def question_check_node(state: AgentState) -> Dict[str, Any]:
    """
    Scan and validate the input question.
    """
    question_status = state["question_status"]
    if sum(question_status) == 0:
        state["question_status"] = []
        return {"question_valid": True, "question_status": []}
    else:
        state["question_status"] = []
        return {"question_valid": False, "question_status": []}


if __name__ == "__main__":
    from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity

    state = {"question": "What is the capital of France?", "question_status": []}

    scan_prompt_injection(state)
    scan_toxicity(state)
    scan_token_limit(state)
