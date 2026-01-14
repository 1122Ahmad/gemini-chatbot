import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

# List models
models = genai.list_models()
for m in models:
    print(m)
