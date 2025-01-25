import os
import sys

from langgraph.graph import END, StateGraph

# Append project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import settings
from src.graph.question_check_node import scan_input_question
from src.graph.retriever_node import retrieve
from src.graph.state import GraphState

workflow = StateGraph(GraphState)
workflow.add_node("scan_question", scan_input_question)
workflow.add_node("retrieve_docs", retrieve)
workflow.add_edge("scan_question", "retrieve_docs")
workflow.add_edge("retrieve_docs", END)

workflow.set_entry_point("scan_question")

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="flow.png")

# Run the workflow
initial_state = {"question": "What is the capital of France?"}
final_state = app.invoke(initial_state)
