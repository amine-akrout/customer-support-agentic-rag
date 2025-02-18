# 🤖 Customer Support Agentic RAG

**An intelligent customer support system leveraging LangGraph and LangChain for Retrieval-Augmented Generation (RAG) with agent-like behavior to deliver accurate, context-aware responses.**

---

## 🚀 Project Overview

This project implements a **sophisticated RAG-based customer support system** that combines the power of **LangGraph for workflow orchestration** and **LangChain for LLM interactions**. The system provides intelligent, context-aware responses to customer queries through a multi-stage validation and retrieval pipeline.

Built with **FastAPI, FAISS, LangGraph, and Ollama**, this system efficiently processes customer support queries while maintaining high accuracy and safety standards through comprehensive validation checks.

---

## ✨ Key Features

✅ **Intelligent Workflow Orchestration** – LangGraph-powered pipeline for sophisticated query processing
✅ **Advanced Document Retrieval** – FAISS vector store for efficient semantic search
✅ **Multi-Stage Validation** – Comprehensive quality checks at each step
✅ **Local LLM Support** – Integration with Ollama for on-premise deployment
✅ **Content Safety** – LLM Guard implementation for safe responses
✅ **Efficient Data Processing** – Polars-based data preprocessing
✅ **API-First Design** – FastAPI backend for scalable deployment

---

## 🏗️ Tech Stack

| Category | Tools Used |
|----------|------------|
| **Programming** | `Python 3.9+` |
| **LLM Integration** | `LangChain`, `Ollama`, `OpenAI API (optional)` |
| **Vector Search** | `FAISS` |
| **Workflow Orchestration** | `LangGraph` |
| **Backend Framework** | `FastAPI` |
| **Data Processing** | `Polars` |
| **Safety & Validation** | `LLM Guard` |
| **Deployment** | `Docker`, `Docker Compose` |

---

## 🔧 System Architecture

The system follows a sophisticated agentic workflow with six main components:

1. **Question Validation**
   - Input safety checks
   - Token limit verification
   - Toxicity detection

2. **Topic Classification**
   - Customer support relevance verification
   - Query categorization

3. **Document Retrieval**
   - FAISS-powered semantic search
   - Context gathering

4. **Document Grading**
   - Relevance scoring
   - Context validation

5. **Answer Generation**
   - Context-aware response generation
   - Local or cloud LLM integration

6. **Answer Validation**
   - Output quality assessment
   - Safety verification

---

## 📁 Project Structure

```
├── data/
│   ├── indexes/          # FAISS index storage
│   └── customer_care_emails.csv
├── src/
│   ├── api/             # FastAPI application
│   ├── graph/           # LangGraph workflow components
│   │   ├── answer_check_node.py
│   │   ├── answer_node.py
│   │   ├── docs_grader_node.py
│   │   ├── graph.py
│   │   ├── question_check_node.py
│   │   ├── retriever_node.py
│   │   ├── state.py
│   │   ├── topic_check_node.py
│   │   └── utils.py
│   ├── static/          # Frontend assets
│   └── indexing/        # Data preprocessing and indexing
├── tests/               # Test cases
├── Dockerfile           # Docker file
└── docker-compose.yml   # Docker configuration
```

---

## ⚙️ Setup & Installation

### 1️⃣ Prerequisites

Before you begin, ensure you have:
- Python 3.9 or higher
- Docker (optional)
- [Ollama](https://ollama.ai/) installed (for local LLM support)
- OpenAI API key (optional, for cloud LLM)

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/amine-akrout/customer-support-agentic-rag.git
cd customer-support-agentic-rag
```

### 3️⃣ Environment Setup

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 4️⃣ Configuration

Create a `.env` file with your settings:
```env
OPENAI_API_KEY=your_api_key_here  # Optional
LANGCHAIN_API_KEY=your_api_key_here # Optional
LANGCHAIN_TRACING_V2= true # Optional
LANGCHAIN_PROJECT=your_project_id_here # Optional

```

---

## 🚀 Usage

### Local Development

1. **Preprocess and Index Data**
```bash
python -m src.indexing.preprocess
```

2. **Start the API Server**
```bash
uvicorn src.main:app --reload
```

3. Access the API at `http://localhost:8000`

### Docker Deployment

1. **Build and Start Containers**
```bash
docker-compose up --build
```

2. Access the API at `http://localhost:8000`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/answer` | Submit question and get response |
| `GET` | `/health` | Check API health status |

Example request:
```json
{
  "question": "How do I return a damaged product?"
}
```

---

## 🔄 Workflow Process

The system follows this process for each query:

1. **Input Processing**
   - Question validation
   - Safety checks
   - Topic classification

2. **Context Retrieval**
   - Document search
   - Relevance scoring
   - Context selection

3. **Response Generation**
   - Answer formulation
   - Quality validation
   - Safety verification

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit (`git commit -am 'Add new feature'`)
5. Push (`git push origin feature/improvement`)
6. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) - LLM framework
- [LangGraph](https://github.com/hwchase17/langgraph) - Workflow orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Ollama](https://ollama.ai/) - Local LLM support

---

## ⭐ Star This Repo!

If you find this project useful, please consider giving it a star! 🌟

---

## 📩 Contact

For questions or feedback, please open an issue in the repository.

---
