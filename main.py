from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import logging
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai
from google.genai import types
import subprocess
import time
import requests

# ---------------- Load .env ----------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")

# ---------------- NGROK CONFIG ----------------
NGROK_PATH = r"C:\Users\Sana\AppData\Local\Microsoft\WindowsApps\ngrok.exe"
NGROK_AUTH_TOKEN = "2wxAoCVbXuAv56L3unqoBln2ljd_7QCAbimd1G5ZpNQ1QXYEg"
APP_PORT = 8000

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Start ngrok ----------------
def start_ngrok():
    try:
        # Add auth token
        subprocess.run(
            [NGROK_PATH, "config", "add-authtoken", NGROK_AUTH_TOKEN],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Start tunnel
        subprocess.Popen(
            [NGROK_PATH, "http", str(APP_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        time.sleep(4)

        # Fetch public URL
        tunnels = requests.get("http://127.0.0.1:4040/api/tunnels").json()
        public_url = tunnels["tunnels"][0]["public_url"]

        logger.info("🚀 NGROK PUBLIC URL: %s", public_url)

    except Exception as e:
        logger.error("❌ Failed to start ngrok: %s", e)

# ---------------- Gemini Model ----------------
GENAI_MODEL = "gemini-2.5-flash"

# ---------------- FastAPI ----------------
app = FastAPI(title="Gemini RAG Chatbot API")

# ---------------- Schemas ----------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)

class ChatResponse(BaseModel):
    reply: str

# ---------------- Load KB Embeddings & FAISS ----------------
logger.info("Loading FAISS index and chunks...")
faiss_index = faiss.read_index("kb.index")

with open("kb_chunks.pkl", "rb") as f:
    kb_chunks = pickle.load(f)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("KB loaded successfully.")

# ---------------- Retrieve Context ----------------
def retrieve_context(query, top_k=3):
    query_vector = embed_model.encode(query).astype("float32")
    D, I = faiss_index.search(np.array([query_vector]), k=top_k)
    return "\n\n".join([kb_chunks[i] for i in I[0]])

# ---------------- Gemini Client ----------------
client = genai.Client(api_key=API_KEY)

# ---------------- Health Check ----------------
@app.get("/")
def root():
    return {"status": "Gemini RAG chatbot backend is running"}

# ---------------- Chat Endpoint ----------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        user_query = request.message
        context = retrieve_context(user_query, top_k=3)

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
                detail="Empty response from AI model"
            )

        return {"reply": response.text}

    except Exception:
        logger.exception("Unexpected server error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# ---------------- Startup Event ----------------
@app.on_event("startup")
def startup_event():
    start_ngrok()
