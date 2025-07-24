"""
RAG System Evaluator
This module evaluates a RAG (Retrieval-Augmented Generation) system using RAGAS metrics.
"""

import random
from typing import Any, Dict, List
from uuid import uuid4

import pandas as pd
from langchain_openai import ChatOpenAI
from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall

from src.config import settings
from src.graph.graph import create_workflow
from src.graph.utils import load_faiss_index


def create_input_scanners() -> List:
    """Create input scanners for content filtering."""
    return [
        PromptInjection(),
        TokenLimit(),
        Toxicity(),
    ]


def create_evaluator_llm() -> LangchainLLMWrapper:
    """Create LLM wrapper for evaluation metrics."""
    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0, max_tokens=1000)
    return LangchainLLMWrapper(llm)


def load_documents_from_faiss(retriever) -> List:
    """
    Retrieve all documents from the FAISS vectorstore.
    """
    retrieved_docs = retriever.vectorstore.docstore._dict.values()
    return list(retrieved_docs)


def sample_documents(
    documents: List, sample_size: int = 10, random_seed: int = 123
) -> List:
    """
    Sample documents for evaluation with reproducible results.
    """
    random.seed(random_seed)
    actual_sample_size = min(sample_size, len(documents))
    return random.sample(documents, actual_sample_size)


def generate_answer(rag_app, query: str, thread_id: str) -> str:
    """
    Generate answer using the RAG system.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = {"question": query}
    result = rag_app.invoke(state, config=config)
    return result


def create_evaluation_record(doc, retriever, rag_app) -> Dict[str, Any]:
    """
    Create a single evaluation record from a document.
    """
    query = doc.metadata.get("question", "")
    reference = doc.metadata.get("answer", "")
    thread_id = str(uuid4())

    # Retrieve relevant documents for the query
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_contexts = [d.metadata for d in retrieved_docs]

    # Generate answer using RAG system
    response = generate_answer(rag_app, query, thread_id)

    return {
        "user_input": query,
        "retrieved_contexts": retrieved_contexts,
        "response": response,
        "reference": reference,
    }


def build_evaluation_dataset(
    documents: List, retriever, rag_app
) -> List[Dict[str, Any]]:
    """
    Build evaluation dataset from sampled documents.
    """
    dataset = []

    for i, doc in enumerate(documents, 1):
        try:
            print(f"Processing document {i}/{len(documents)}...")
            record = create_evaluation_record(doc, retriever, rag_app)
            dataset.append(record)
        except Exception as e:
            print(f"Error processing document {i}: {e}")
            continue

    return dataset


def get_evaluation_metrics() -> List:
    """Get list of RAGAS evaluation metrics."""
    return [LLMContextRecall(), Faithfulness(), FactualCorrectness()]


def run_evaluation(dataset: List[Dict[str, Any]], evaluator_llm: LangchainLLMWrapper):
    """
    Run RAGAS evaluation on the dataset.

    Args:
        dataset: List of evaluation records
        evaluator_llm: LLM wrapper for evaluation

    Returns:
        Evaluation results
    """
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    metrics = get_evaluation_metrics()

    print("Running evaluation with RAGAS metrics...")
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )

    return results


def display_results(dataset: List[Dict[str, Any]], results):
    """
    Display evaluation results in a readable format.
    """
    pd.set_option("display.max_colwidth", None)

    print("=== Dataset Overview ===")
    df = pd.DataFrame(dataset)
    print(f"Total samples: {len(dataset)}")
    print(f"Columns: {list(df.columns)}")
    print("\n=== Sample Data ===")
    print(df.head())

    print("\n=== Evaluation Results ===")
    print(results)


def main():
    """Main evaluation pipeline."""
    print("Starting RAG evaluation...")

    # Initialize components
    print("Loading components...")
    input_scanners = create_input_scanners()
    retriever = load_faiss_index()
    rag_app = create_workflow(retriever, input_scanners=input_scanners)
    evaluator_llm = create_evaluator_llm()

    # Load and sample documents
    print("Loading documents...")
    documents = load_documents_from_faiss(retriever)
    print(f"Total documents available: {len(documents)}")

    sampled_docs = sample_documents(documents, sample_size=10, random_seed=123)
    print(f"Sampled {len(sampled_docs)} documents for evaluation")

    # Build evaluation dataset
    print("Building evaluation dataset...")
    dataset = build_evaluation_dataset(sampled_docs, retriever, rag_app)
    print(f"Successfully created {len(dataset)} evaluation records")

    # Run evaluation
    results = run_evaluation(dataset, evaluator_llm)

    # Display results
    display_results(dataset, results)

    print("Evaluation completed!")


if __name__ == "__main__":
    main()
