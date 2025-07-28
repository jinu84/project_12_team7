# streamlit_basic_chat_gpt4o.py

import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# OpenAI API 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="💬 GPT-4o 기본 챗봇")
st.title("💬 GPT-4o와 대화해보세요")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 사용자 입력
user_input = st.chat_input("무엇이든 물어보세요")

if user_input:
    with st.spinner("GPT-4o가 답변 생성 중..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 유용하고 친절한 AI입니다. 한국어로 대답해주세요."},
                *[{"role": "user", "content": q} for q, _ in st.session_state.chat_history],
                *[{"role": "assistant", "content": a} for _, a in st.session_state.chat_history],
                {"role": "user", "content": user_input},
            ]
        )
        answer = response.choices[0].message.content
        st.session_state.chat_history.append((user_input, answer))

# 채팅 출력
for q, a in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
