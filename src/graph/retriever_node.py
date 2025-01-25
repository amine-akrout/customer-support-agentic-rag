from typing import Any, Dict

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config import settings
from src.graph.state import GraphState


def load_faiss_index() -> FAISS:
    """
    Load the FAISS index.

    Returns:
        FAISS retriever object.
    """
    try:
        logger.info("Loading FAISS index...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDINGS_MODEL_NAME
        )
        vector_store = FAISS.load_local(
            settings.FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        logger.exception("Failed to load FAISS index.")
        raise e

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.FAISS_TOP_K},
    )

    return retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents from the FAISS index.

    Args:
        state (GraphState): The graph state.

    Returns:
        Dict[str, Any]: The updated graph state.
    """
    # Load the FAISS index
    faiss_retriever = load_faiss_index()

    # Retrieve the question
    question = state["question"]

    # Retrieve documents
    documents = faiss_retriever.invoke(question)

    # extracted metadata from the documents
    metadata = [doc.metadata for doc in documents]

    # Update the graph state
    state["documents"] = metadata

    return state
