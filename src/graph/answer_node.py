from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings
from src.graph.state import AgentState


def generate_answer(state: AgentState):
    llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )
    question = state["question"]
    context = state["documents"]

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template=template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context})
    state["llm_output"] = result
    state["prompt"] = prompt.format(question=question, context=context)
    return state
