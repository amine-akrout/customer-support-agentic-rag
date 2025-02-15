"""
Preprocess the dataset and create a FAISS index.
"""

import os

import polars as pl
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

# local imports
from src.config import settings


# Load the dataset using Polars
def download_and_preprocess_dataset() -> pl.DataFrame:
    """Download and preprocess the dataset using Polars."""
    # Load the dataset
    customer_care_df = pl.read_csv(settings.DATA_URL)
    logger.info(f"Loaded dataset with {customer_care_df.height} records.")

    # Preprocess the dataset
    customer_care_df = customer_care_df.select(["instruction", "response"]).rename(
        {"instruction": "question", "response": "answer"}
    )
    customer_care_df = customer_care_df.drop_nulls()
    logger.info(f"Preprocessed dataset with {customer_care_df.height} records.")

    return customer_care_df


def generate_documents(customer_care_df: pl.DataFrame) -> list[Document]:
    """Generate documents from a Polars DataFrame."""
    documents = [
        Document(
            page_content=row["question"],
            metadata=row,
            id=idx,
        )
        for idx, row in enumerate(customer_care_df.to_dicts())
    ]
    logger.info(f"Generated {len(documents)} documents.")
    return documents


def create_faiss_index(documents: list[Document]) -> None:
    """Create or update FAISS index, avoiding duplicates."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL_NAME)
    index_path = settings.FAISS_INDEX_PATH

    if os.path.exists(index_path):
        # Load existing index
        logger.info("Loading existing FAISS index...")
        faiss_index = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        # Get existing document IDs
        existing_ids = set(faiss_index.index_to_docstore_id.values())
        # Filter new documents
        new_docs = [doc for doc in documents if doc.id not in existing_ids]
        if new_docs:
            logger.info(f"Adding {len(new_docs)} new documents.")
            faiss_index.add_documents(new_docs)
            faiss_index.save_local(index_path)
            logger.info(f"Updated index saved to {index_path}")
        else:
            logger.info("No new documents to add.")
    else:
        # Create new index
        logger.info("Creating new FAISS index...")
        faiss_index = FAISS.from_documents(documents, embeddings)
        faiss_index.save_local(index_path)
        logger.info(f"New index saved to {index_path}")


def embed_and_index():
    """Embed and index the dataset."""
    # Download and preprocess the dataset
    customer_care_df = download_and_preprocess_dataset()

    # Generate documents
    documents = generate_documents(customer_care_df)

    # Create or update the FAISS index
    create_faiss_index(documents)


if __name__ == "__main__":
    embed_and_index()
