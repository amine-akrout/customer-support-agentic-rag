from typing import Any, Dict

from src.graph.state import AgentState


def retrieve(state: AgentState, faiss_retriever) -> Dict[str, Any]:
    """
    Retrieve documents from the FAISS index.

    Args:
        state (AgentState): The graph state.

    Returns:
        Dict[str, Any]: The updated graph state.
    """
    # Load the FAISS index
    # faiss_retriever = load_faiss_index()

    # Retrieve the question
    question = state["question"]

    # Retrieve documents
    documents = faiss_retriever.invoke(question)

    # extracted metadata from the documents
    metadata = [doc.metadata for doc in documents]

    # Update the graph state
    state["documents"] = metadata

    return state
