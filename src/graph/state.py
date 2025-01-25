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
    on_topic: bool
        The topic status.
    generation: str
        The LLM generation.
    documents: List[str]
        The retrieved documents.

    """

    question: str
    question_status: str
    on_topic: bool
    generation: str
    documents: List[str]
