import os
import sys
from functools import partial

from langgraph.graph import END, StateGraph

# Append project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import settings
from src.graph.answer_check_node import scan_output_answer
from src.graph.answer_node import answer_node
from src.graph.docs_grader_node import grade_documents_node
from src.graph.question_check_node import scan_input_question
from src.graph.retriever_node import load_faiss_index, retrieve
from src.graph.state import AgentState
from src.graph.topic_check_node import topic_classifier


def create_workflow(retriever):
    """Create a workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("scan_question", scan_input_question)
    workflow.add_node("topic_classifier", topic_classifier)
    workflow.add_node("retrieve_docs", partial(retrieve, faiss_retriever=retriever))
    workflow.add_node("docs_grader", grade_documents_node)
    workflow.add_node("generate_answer", answer_node)
    workflow.add_node("check_answer", scan_output_answer)
    workflow.add_conditional_edges(
        "scan_question",
        lambda state: state["question_status"],
        {
            "valid": "topic_classifier",
            "invalid": END,
        },
    )
    workflow.add_conditional_edges(
        "topic_classifier",
        lambda state: state["on_topic"],
        {
            "Yes": "retrieve_docs",
            "No": END,
        },
    )
    workflow.add_edge("retrieve_docs", "docs_grader")
    workflow.add_edge("docs_grader", "generate_answer")
    workflow.add_edge("generate_answer", "check_answer")
    workflow.add_edge("check_answer", END)
    workflow.set_entry_point("scan_question")
    return workflow


if __name__ == "__main__":
    # Load the FAISS index
    faiss_retriever = load_faiss_index()

    workflow = create_workflow(faiss_retriever)
    app = workflow.compile()
    app.get_graph().draw_mermaid_png(output_file_path="flow.png")

    # Run the workflow
    state1 = {"question": "What is the capital of France?"}
    state2 = {"question": "I wnat to return a package"}
    final_state1 = app.invoke(state1)
    final_state2 = app.invoke(state2)
