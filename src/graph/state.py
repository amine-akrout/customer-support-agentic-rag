from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Graph state.

    Attributes:
    -----------
    question: str
        The question.
    generation: str
        The LLM generation.
    documents: List[str]
        The retrieved documents.

    """

    question: str
    generation: str
    documents: List[str]
