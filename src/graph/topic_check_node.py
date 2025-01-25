from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.config import settings
from src.graph.state import AgentState


# Question Classifier
class GradeTopic(BaseModel):
    """Boolean value to check whether a question is related to the customer support topic"""

    score: str = Field(
        description="Question is about customer support? If yes -> 'Yes' if not -> 'No'"
    )


def topic_classifier(state: AgentState):
    question = state["question"]

    system = """You are a grader assessing the relevance of a retrieved document to a user question. 
        Only answer if the question is about customer support topic.

        If the question is about customer support topic response with "Yes", otherwise respond with "No".
        """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}"),
        ]
    )
    llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )
    structured_llm = llm.with_structured_output(GradeTopic)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({"question": question})
    state["on_topic"] = result.score
    return state
