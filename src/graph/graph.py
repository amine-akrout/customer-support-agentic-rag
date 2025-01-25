import os
import sys

from langgraph.graph import END, StateGraph

# Append project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import settings
from src.graph.answer_node import generate_answer
from src.graph.question_check_node import scan_input_question
from src.graph.retriever_node import retrieve
from src.graph.state import AgentState
from src.graph.topic_check_node import topic_classifier

workflow = StateGraph(AgentState)
workflow.add_node("scan_question", scan_input_question)
workflow.add_node("retrieve_docs", retrieve)
workflow.add_node("topic_classifier", topic_classifier)
workflow.add_node("generate_answer", generate_answer)
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
workflow.add_edge("retrieve_docs", "generate_answer")

workflow.add_edge("generate_answer", END)

workflow.set_entry_point("scan_question")

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="flow.png")

# Run the workflow
initial_state = {"question": "What is the capital of France?"}
final_state = app.invoke(initial_state)
