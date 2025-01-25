from typing import List, TypedDict


class AgentState(TypedDict):
    """
    Graph state.

    Attributes:
    -----------
    question: str
        The question.
    question_status: str
        The question status.
    generation: str
        The LLM generation.
    documents: List[str]
        The retrieved documents.

    """

    question: str
    question_status: str
    generation: str
    documents: List[str]
