from typing import Any, Dict

import torch
import torch._inductor.config
from llm_guard import scan_output
from llm_guard.output_scanners import LanguageSame, Relevance, Sentiment

from src.graph.state import AgentState

torch.set_float32_matmul_precision("high")
torch._inductor.config.fx_graph_cache = True

language_same_scanner = LanguageSame(use_onnx=True)
relevance_scanner = Relevance(use_onnx=True)
sentiment_scanner = Sentiment()


def check_language_same(state: AgentState) -> Dict[str, Any]:
    """Run LanguageSame check."""
    output = state["llm_output"]
    prompt = state["prompt"]
    _, results_valid, _ = scan_output(
        scanners=[language_same_scanner], output=output, prompt=prompt
    )
    same_language = not results_valid.get("LanguageSame", True)
    return {"answer_status": [1 if same_language else 0]}


def check_relevance(state: AgentState) -> Dict[str, Any]:
    """Run Relevance check"""
    output = state["llm_output"]
    prompt = state["prompt"]
    _, results_valid, _ = scan_output(
        scanners=[relevance_scanner], output=output, prompt=prompt
    )
    relevant_answer = not results_valid.get("Relevance", True)
    return {"answer_status": [1 if relevant_answer else 0]}


def check_sentiment(state: AgentState) -> Dict[str, Any]:
    """Run Sentiment check"""
    output = state["llm_output"]
    prompt = state["prompt"]
    _, results_valid, _ = scan_output(
        scanners=[sentiment_scanner], output=output, prompt=prompt
    )
    sentiment = not results_valid.get("Sentiment", True)
    return {"answer_status": [1 if sentiment else 0]}


def answer_check_node(state: AgentState) -> Dict[str, Any]:
    """Run all answer checks"""
    answer_status = state["answer_status"]
    answer = state["llm_output"]
    all_checks_passed = all(status == 0 for status in answer_status[-3:])
    if all_checks_passed:
        state["answer_valid"] = True
        return {"llm_output": answer}
    state["answer_valid"] = False
    return {"llm_output": "Answer failed checks, please try again."}


if __name__ == "__main__":
    state = {"llm_output": "Hello, how can I assist you today?", "prompt": "Hello"}
    check_language_same(state)

    state2 = {
        "llm_output": "Bonjour, comment puis-je vous aider aujourd'hui?",
        "prompt": "Hello",
    }
    check_language_same(state2)

    # check relevance with non relevant example
    state3 = {
        "llm_output": "Hello, how can I assist you today?",
        "prompt": "What is the weather today?",
    }
    check_relevance(state3)

    # check relevance with relevant example
    state4 = {
        "llm_output": "The weather today is sunny.",
        "prompt": "What is the weather today?",
    }
    check_relevance(state4)

    # check toxicity with toxic example
    state5 = {"llm_output": "You are an idiot.", "prompt": "What is the weather today?"}
    check_sentiment(state5)

    # check toxicity with non toxic example
    state6 = {
        "llm_output": "Hello, how can I assist you today?",
        "prompt": "What is the weather today?",
    }
    check_sentiment(state6)
