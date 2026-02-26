import logging
import os
import subprocess
import time

import requests
import streamlit as st

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Environment config ----------------
NGROK_PATH = os.getenv("NGROK_PATH", "ngrok")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
ENABLE_NGROK = os.getenv("ENABLE_NGROK", "false").lower() == "true"

BACKEND_ROOT = os.getenv("BACKEND_ROOT", "http://127.0.0.1:8000")
BACKEND_URL = os.getenv("BACKEND_URL", f"{BACKEND_ROOT}/chat")
HEALTH_URL = f"{BACKEND_ROOT}/health"


def start_ngrok_frontend() -> None:
    if not ENABLE_NGROK:
        st.sidebar.info("ngrok disabled. Set ENABLE_NGROK=true if you need a public link.")
        return

    if not NGROK_AUTH_TOKEN:
        st.sidebar.warning("ENABLE_NGROK=true but NGROK_AUTH_TOKEN is missing.")
        return

    try:
        subprocess.run(
            [NGROK_PATH, "config", "add-authtoken", NGROK_AUTH_TOKEN],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        subprocess.Popen(
            [NGROK_PATH, "http", str(STREAMLIT_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(4)
        tunnels = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=5).json()
        public_url = tunnels["tunnels"][0]["public_url"]

        logger.info("Frontend ngrok URL: %s", public_url)
        st.sidebar.success(f"Public URL:\n{public_url}")

    except Exception as exc:
        logger.error("Failed to start frontend ngrok: %s", exc)
        st.sidebar.error("Failed to start ngrok")


if "ngrok_started" not in st.session_state:
    start_ngrok_frontend()
    st.session_state.ngrok_started = True

st.set_page_config(page_title="THM Chatbot", page_icon="🤖")
st.title("🤖 THM Chatbot")
st.caption("Gemini + RAG support assistant with source snippets and latency metadata.")

with st.sidebar:
    st.subheader("Project highlights")
    st.markdown("- Gemini generation\n- FAISS retrieval\n- Streamlit chat UX\n- FastAPI backend")
    try:
        health = requests.get(HEALTH_URL, timeout=5)
        health.raise_for_status()
        payload = health.json()
        st.success(f"Backend OK | model: {payload.get('model')} | chunks: {payload.get('kb_chunks')}")
    except requests.RequestException:
        st.warning("Backend health check unavailable")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])
        if msg.get("sources"):
            with st.expander("Retrieved sources"):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"**Source {i}:** {src}")

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = requests.post(BACKEND_URL, json={"message": user_input}, timeout=45)
        response.raise_for_status()
        payload = response.json()
        bot_reply = payload.get("reply", "No reply from backend")
        meta = f"Model: {payload.get('model', 'n/a')} | Latency: {payload.get('latency_ms', 'n/a')} ms"
        sources = payload.get("sources", [])
    except requests.exceptions.RequestException as exc:
        bot_reply = f"Backend error: {exc}"
        meta = ""
        sources = []

    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply, "meta": meta, "sources": sources}
    )
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
        if meta:
            st.caption(meta)
        if sources:
            with st.expander("Retrieved sources"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**Source {i}:** {src}")
