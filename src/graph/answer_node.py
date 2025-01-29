from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import settings
from src.graph.state import AgentState


def generate_answer(question: str, context: list, local_llm: bool = True):
    """
    Generate answer to the question based on the context.
    """
    if local_llm:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    else:
        llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
        )
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template=template)
    formatted_prompt = prompt.format(question=question, context=context)

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context})
    return result, formatted_prompt


def answer_node(state: AgentState):
    """Generate answer node"""
    question = state["question"]
    context = state["documents"]
    answer, prompt = generate_answer(question, context)
    state["llm_output"] = answer
    state["prompt"] = prompt
    return state
