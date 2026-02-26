# Gemini RAG Chatbot (FastAPI + Streamlit)

A portfolio-ready customer support chatbot using:
- **Gemini 2.5 Flash** for response generation
- **FAISS + SentenceTransformers** for retrieval-augmented generation (RAG)
- **FastAPI** backend API
- **Streamlit** chat frontend

This project simulates a real support assistant for **My Talk Home** with empathetic, step-by-step responses grounded in a knowledge base.

## Why this project is strong

- Demonstrates **end-to-end AI app delivery** (API, retrieval, LLM integration, UI).
- Includes **client-friendly trust features**:
  - source snippets shown with each reply
  - latency and model metadata
  - backend health endpoint
- Shows **production-minded engineering**:
  - environment-driven configuration
  - optional public demo with ngrok
  - Docker and docker-compose support

## Architecture

1. User sends a message in Streamlit UI.
2. FastAPI receives request at `/chat`.
3. Query embedding is generated with `all-MiniLM-L6-v2`.
4. FAISS retrieves top relevant chunks from the KB.
5. Gemini generates final response from retrieved context.
6. API returns answer + source snippets + latency.

## Quickstart (local)

### 1) Install dependencies

```bash
pip install -r requirements.txt
pip install -r frontend/requirements.txt
```

### 2) Configure environment

```bash
cp .env.example .env
```

Set at least:
- `GEMINI_API_KEY`

### 3) Run backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4) Run frontend

```bash
streamlit run frontend/app.py --server.port 8501
```

## Quickstart (Docker)

```bash
docker compose up --build
```

- Backend: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

## API Endpoints

- `GET /` - basic status
- `GET /health` - model + KB health information
- `POST /chat` - response payload with `reply`, `sources`, `model`, and `latency_ms`


## Project structure

```text
main.py                 # FastAPI backend + RAG pipeline
frontend/app.py         # Streamlit chat UI
kb.index                # FAISS vector index
kb_chunks.pkl           # Retrieved text chunks
THM_kb_with_text.csv    # Source KB dataset
Dockerfile              # Backend container
frontend/Dockerfile     # Frontend container
docker-compose.yml      # One-command multi-service run
```
