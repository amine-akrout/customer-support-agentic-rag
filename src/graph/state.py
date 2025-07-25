from operator import add
from typing import Annotated, List, TypedDict


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
    prompt: str
        The prompt.
    llm_output: str
        The LLM generation.
    documents: List[str]
        The retrieved documents.
    answer_status: str
        The answer status.

    """

    # question: Annotated[str, add]
    question: str
    question_status: Annotated[list, add]
    question_valid: bool
    on_topic: bool
    prompt: str
    llm_output: str
    documents: List[str]
    answer_status: str
