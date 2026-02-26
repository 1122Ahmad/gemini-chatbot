import logging
import os
import pickle
import subprocess
import time
from typing import List

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ---------------- Load environment ----------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")

NGROK_PATH = os.getenv("NGROK_PATH", "ngrok")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
ENABLE_NGROK = os.getenv("ENABLE_NGROK", "false").lower() == "true"
GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-2.5-flash")
TOP_K = int(os.getenv("TOP_K", "3"))

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- FastAPI ----------------
app = FastAPI(
    title="Gemini RAG Chatbot API",
    description="Portfolio-ready customer support chatbot with Gemini + FAISS retrieval.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    reply: str
    sources: List[str] = Field(default_factory=list)
    model: str
    latency_ms: int


# ---------------- Load KB Embeddings & FAISS ----------------
logger.info("Loading FAISS index and chunks...")
faiss_index = faiss.read_index("kb.index")

with open("kb_chunks.pkl", "rb") as f:
    kb_chunks = pickle.load(f)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("KB loaded successfully.")


# ---------------- Retrieve Context ----------------
def retrieve_context(query: str, top_k: int = TOP_K) -> List[str]:
    query_vector = embed_model.encode(query).astype("float32")
    _, indices = faiss_index.search(np.array([query_vector]), k=top_k)
    return [kb_chunks[i] for i in indices[0] if 0 <= i < len(kb_chunks)]


# ---------------- Gemini Client ----------------
client = genai.Client(api_key=API_KEY)


# ---------------- ngrok ----------------
def start_ngrok() -> None:
    if not ENABLE_NGROK:
        logger.info("ngrok startup disabled. Set ENABLE_NGROK=true to enable public tunnel.")
        return

    if not NGROK_AUTH_TOKEN:
        logger.warning("ENABLE_NGROK=true but NGROK_AUTH_TOKEN is missing; skipping ngrok startup.")
        return

    try:
        subprocess.run(
            [NGROK_PATH, "config", "add-authtoken", NGROK_AUTH_TOKEN],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        subprocess.Popen(
            [NGROK_PATH, "http", str(APP_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(4)
        tunnels = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=5).json()
        public_url = tunnels["tunnels"][0]["public_url"]
        logger.info("Backend ngrok URL: %s", public_url)
    except Exception as exc:
        logger.error("Failed to start ngrok: %s", exc)


# ---------------- Health Check ----------------
@app.get("/")
def root():
    return {"status": "Gemini RAG chatbot backend is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": GENAI_MODEL,
        "kb_chunks": len(kb_chunks),
        "top_k": TOP_K,
    }


# ---------------- Chat Endpoint ----------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        started = time.perf_counter()
        user_query = request.message
        source_chunks = retrieve_context(user_query)
        context = "\n\n".join(source_chunks)

        system_prompt = (
            "You are a warm, professional British support agent for My Talk Home. "
            "Greet the customer politely, acknowledge their concern and emotional state, "
            "and respond empathetically. Give clear, step-by-step instructions strictly "
            "based on My Talk Home information. Use simple British English. "
            "If unresolved, apologise and suggest a live agent."
        )

        response = client.models.generate_content(
            model=GENAI_MODEL,
            contents=f"Context:\n{context}\n\nQuestion:\n{user_query}",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )

        if not response or not getattr(response, "text", None):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Empty response from AI model",
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        preview_sources = [chunk[:220] + ("..." if len(chunk) > 220 else "") for chunk in source_chunks]

        return {
            "reply": response.text,
            "sources": preview_sources,
            "model": GENAI_MODEL,
            "latency_ms": latency_ms,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected server error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@app.on_event("startup")
def startup_event():
    start_ngrok()
