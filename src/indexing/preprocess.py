import json
import os
import pickle
import sys
from typing import Optional

import pandas as pd
from datasets import load_dataset
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import settings


# Load the dataset
def download_and_preprocess_dataset():
    """Download and preprocess the dataset."""
    # Load the dataset
    customer_care_df = pd.read_csv(settings.DATA_URL)
    logger.info(f"Loaded dataset with {len(customer_care_df)} records.")

    # Preprocess the dataset
    customer_care_df = customer_care_df[["instruction", "response"]]
    customer_care_df.columns = ["question", "answer"]
    customer_care_df.dropna(inplace=True)
    customer_care_df.reset_index(drop=True, inplace=True)
    logger.info(f"Preprocessed dataset with {len(customer_care_df)} records.")

    return customer_care_df


def generate_documents(customer_care_df: pd.DataFrame) -> list[Document]:
    """Generate documents from a DataFrame."""
    documents = [
        Document(
            page_content=row["question"],
            metadata=row.to_dict(),
            id=idx,
        )
        for idx, row in customer_care_df.iterrows()
    ]
    logger.info(f"Generated {len(documents)} documents.")
    return documents


def create_faiss_index(documents: list) -> None:
    """Creates and saves a FAISS index."""
    # Load the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL_NAME)
    try:
        logger.info("Creating FAISS index...")
        faiss_index = FAISS.from_documents(documents, embeddings)
        faiss_index.save_local(settings.FAISS_INDEX_PATH)
        logger.info(f"FAISS index saved at {settings.FAISS_INDEX_PATH}")
    except Exception as e:
        logger.exception("Failed to create FAISS index.")
        raise e


def embed_and_index():
    """Embed and index the dataset."""
    # Download and preprocess the dataset
    customer_care_df = download_and_preprocess_dataset()

    # Generate documents
    documents = generate_documents(customer_care_df)

    # Create the FAISS index
    create_faiss_index(documents)


if __name__ == "__main__":
    embed_and_index()
