"""
RAG System Evaluator

This module evaluates a RAG system using RAGAS metrics by sampling documents,
generating answers, and measuring performance.
"""

import os
import random
from uuid import uuid4

from langchain_openai import ChatOpenAI
from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity
from loguru import logger
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall

from src.config import settings
from src.graph.graph import create_workflow
from src.graph.utils import load_faiss_index


def setup_components():
    """Initialize all required components for RAG evaluation."""
    input_scanners = [PromptInjection(), TokenLimit(), Toxicity()]
    retriever = load_faiss_index()
    rag_app = create_workflow(retriever, input_scanners=input_scanners)

    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0, max_tokens=1000)
    evaluator_llm = LangchainLLMWrapper(llm)

    return retriever, rag_app, evaluator_llm


def prepare_evaluation_data(retriever, rag_app):
    """
    Sample documents and create evaluation dataset.
    """
    # Load and sample documents
    sample_size = settings.EVALUATION_SAMPLE_SIZE
    random.seed(settings.EVALUATION_RANDOM_SEED)
    documents = list(retriever.vectorstore.docstore._dict.values())
    sampled_docs = random.sample(documents, min(sample_size, len(documents)))

    logger.info(f"Processing {len(sampled_docs)} documents...")

    # Build evaluation dataset
    dataset = []
    for i, doc in enumerate(sampled_docs, 1):
        try:
            query = doc.metadata.get("question", "")
            reference = doc.metadata.get("answer", "")
            thread_id = str(uuid4())

            # Get relevant documents and generate answer
            retrieved_docs = retriever.get_relevant_documents(query)
            retrieved_contexts = [d.metadata for d in retrieved_docs]

            cleaned_contexts = [
                f"question: {d['question']}\nanswer: {d['answer']}"
                for d in retrieved_contexts
            ]

            config = {"configurable": {"thread_id": thread_id}}
            state = {"question": query}
            response = rag_app.invoke(state, config=config)
            response = response.get("llm_output", "")

            dataset.append(
                {
                    "user_input": query,
                    "retrieved_contexts": cleaned_contexts,
                    "response": response,
                    "reference": reference,
                }
            )

            logger.info(f"Processed {i}/{len(sampled_docs)}")

        except Exception as e:
            logger.info(f"Error processing document {i}: {e}")
            continue

    return dataset


def run_evaluation(dataset, evaluator_llm):
    """Run RAGAS evaluation and display results."""
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]

    logger.info("Running RAGAS evaluation...")
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )

    # Convert results to DataFrame for better display
    output_dir = settings.EVALUATION_OUTPUT_DIR
    results_df = results.to_pandas()
    # Save evaluation results
    results_html_path = os.path.join(output_dir, "evaluation_results.html")
    results_df.to_html(results_html_path, index=False)
    mean_scores = results_df.mean(numeric_only=True).round(4).to_frame(name="score")
    mean_scores_path = os.path.join(output_dir, "mean_scores.html")
    mean_scores.to_html(mean_scores_path)
    logger.info(f"Evaluation results saved to {output_dir}")


def main():
    """Main evaluation pipeline."""
    logger.info("Starting RAG evaluation...")

    # Setup
    retriever, rag_app, evaluator_llm = setup_components()

    # Prepare data and evaluate
    dataset = prepare_evaluation_data(retriever, rag_app)
    run_evaluation(dataset, evaluator_llm)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
