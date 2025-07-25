"""
Graph-based workflow for the assistant.
"""

from functools import partial

from langchain.globals import set_debug
from langgraph.graph import END, START, StateGraph

# local imports
from src.graph.answer_check_node import (
    answer_check_node,
    check_language_same,
    check_relevance,
    check_sentiment,
)
from src.graph.answer_node import answer_node
from src.graph.docs_grader_node import grade_documents_node
from src.graph.question_check_node import (
    question_check_node,
    scan_prompt_injection,
    scan_token_limit,
    scan_toxicity,
)
from src.graph.retriever_node import retrieve
from src.graph.state import AgentState
from src.graph.topic_check_node import topic_classifier
from src.graph.utils import load_faiss_index

set_debug(True)


def create_workflow(retriever):
    """Create a workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node(
        "scan_prompt_injection",
        scan_prompt_injection,
    )
    workflow.add_node(
        "scan_toxicity",
        scan_toxicity,
    )
    workflow.add_node(
        "scan_token_limit",
        scan_token_limit,
    )
    workflow.add_node("question_check_node", question_check_node)
    workflow.add_conditional_edges(
        "question_check_node",
        lambda state: state["question_valid"],
        {True: "topic_classifier", False: END},
    )
    workflow.add_node("topic_classifier", topic_classifier)
    workflow.add_conditional_edges(
        "topic_classifier",
        lambda state: state["on_topic"],
        {
            "Yes": "retrieve_docs",
            "No": END,
        },
    )
    workflow.add_node("retrieve_docs", partial(retrieve, faiss_retriever=retriever))
    workflow.add_node("docs_grader", grade_documents_node)
    workflow.add_node("check_language_same", check_language_same)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("check_sentiment", check_sentiment)
    workflow.add_node("answer_check_node", answer_check_node)

    workflow.add_node("generate_answer", answer_node)

    workflow.add_edge(START, "scan_prompt_injection")
    workflow.add_edge(START, "scan_toxicity")
    workflow.add_edge(START, "scan_token_limit")
    workflow.add_edge("scan_prompt_injection", "question_check_node")
    workflow.add_edge("scan_toxicity", "question_check_node")
    workflow.add_edge("scan_token_limit", "question_check_node")
    workflow.add_edge("retrieve_docs", "docs_grader")
    workflow.add_edge("docs_grader", "generate_answer")
    workflow.add_edge("generate_answer", "check_language_same")
    workflow.add_edge("generate_answer", "check_relevance")
    workflow.add_edge("generate_answer", "check_sentiment")
    workflow.add_edge("check_language_same", "answer_check_node")
    workflow.add_edge("check_relevance", "answer_check_node")
    workflow.add_edge("check_sentiment", "answer_check_node")
    workflow.add_edge("answer_check_node", END)

    graph = workflow.compile()
    return graph


if __name__ == "__main__":
    # Load the FAISS index
    faiss_retriever = load_faiss_index()

    app = create_workflow(faiss_retriever)
    app.get_graph().draw_mermaid_png(output_file_path="flow.png")

    # Run the workflow

    config = {"configurable": {"thread_id": 1}}

    state1 = {"question": "What is the capital of France?"}
    state2 = {"question": "I wnat to return a package"}
    final_state1 = app.invoke(state1, config=config)
    final_state2 = app.invoke(state2, config=config)
