import os
import sys
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import settings
from src.graph.state import AgentState


# Question Classifier
class GradeTopic(BaseModel):
    """Boolean value to check whether a question is related to the customer support topic"""

    score: str = Field(
        description="Question is about customer support? If yes -> 'Yes' if not -> 'No'"
    )


def classify_topic(question: str, local_llm: bool = True) -> Dict[str, Any]:
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
    if local_llm:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        grader_llm = grade_prompt | llm
        grader_output = grader_llm.invoke({"question": question})
        result = grader_output.content
        return result

    else:
        llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
        )
    structured_llm = llm.with_structured_output(GradeTopic)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({"question": question})
    return result.score


def topic_classifier(state: AgentState):
    """Classify the topic of the question"""
    question = state["question"]
    result = classify_topic(question)
    state["on_topic"] = result
    state["llm_output"] = (
        "Please ask a question about customer support so I can help you better."
    )
    return state
