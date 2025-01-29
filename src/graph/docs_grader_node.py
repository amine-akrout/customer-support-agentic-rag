from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.graph.state import AgentState


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def retrieval_grader(doc: str, question: str, local_llm: bool = True):
    """Create a grader to check the relevance of the retrieved documents."""

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )
    if local_llm:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        retrieval_grader = grade_prompt | llm
        grader_output = retrieval_grader.invoke({"question": question, "document": doc})
        return grader_output.content
    else:
        llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
        )
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        retrieval_grader = grade_prompt | structured_llm_grader
        grader_output = retrieval_grader.invoke({"question": question, "document": doc})
    return grader_output.score


def grade_documents_node(state: AgentState):
    docs = state["documents"]
    question = state["question"]
    filtered_docs = []
    for doc in docs:
        grade = retrieval_grader(doc, question)
        if grade.lower() == "yes":
            filtered_docs.append(doc)
    state["documents"] = filtered_docs
    return state
