import streamlit as st
import requests
import subprocess
import time
import logging
import os
import requests as req

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- NGROK CONFIG ----------------
NGROK_PATH = r"C:\Users\Sana\AppData\Local\Microsoft\WindowsApps\ngrok.exe"
NGROK_AUTH_TOKEN = "2wxAoCVbXuAv56L3unqoBln2ljd_7QCAbimd1G5ZpNQ1QXYEg"
STREAMLIT_PORT = 8501

# ---------------- BACKEND API ----------------
# Update this AFTER backend ngrok starts
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000/chat"
)

# ---------------- Start ngrok for Streamlit ----------------
def start_ngrok_frontend():
    try:
        # Add auth token (safe if already added)
        subprocess.run(
            [NGROK_PATH, "config", "add-authtoken", NGROK_AUTH_TOKEN],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        # Start ngrok tunnel
        subprocess.Popen(
            [NGROK_PATH, "http", str(STREAMLIT_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(4)

        # Fetch public URL
        tunnels = req.get("http://127.0.0.1:4040/api/tunnels").json()
        public_url = tunnels["tunnels"][0]["public_url"]

        logger.info("🌍 STREAMLIT NGROK URL: %s", public_url)

        st.sidebar.success(f"🌍 Public URL:\n{public_url}")

    except Exception as e:
        logger.error("Failed to start frontend ngrok: %s", e)
        st.sidebar.error("❌ Failed to start ngrok")

# ---------------- Start ngrok ONCE ----------------
if "ngrok_started" not in st.session_state:
    start_ngrok_frontend()
    st.session_state.ngrok_started = True

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="THM Chatbot", page_icon="🤖")
st.title("🤖 THM Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = requests.post(
            BACKEND_URL,
            json={"message": user_input},
            timeout=30
        )
        response.raise_for_status()
        bot_reply = response.json().get("reply", "⚠️ No reply from backend")

    except requests.exceptions.RequestException as e:
        bot_reply = f"⚠️ Backend error: {e}"

    # Show bot reply
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
