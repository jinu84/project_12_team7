# app.py 또는 chatbot_dashboard.py 상단
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="./dash/.env")  # 경로 정확히 지정

print("🔑 ENV VALUE:", os.getenv("OPENAI_API_KEY"))
