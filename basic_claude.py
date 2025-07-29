# streamlit_claude_chat.py

import streamlit as st
import anthropic
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Anthropic 클라이언트 초기화
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

st.set_page_config(page_title="💬 Claude 기본 챗봇")
st.title("💬 Claude 3와 대화해보세요")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 사용자 입력
user_input = st.chat_input("무엇이든 물어보세요")

if user_input:
    with st.spinner("Claude가 답변 생성 중..."):

        # 과거 대화 기록을 통합 메시지로 구성
        chat_messages = ""
        for q, a in st.session_state.chat_history:
            chat_messages += f"\n\nHuman: {q}\n\nAssistant: {a}"
        chat_messages += f"\n\nHuman: {user_input}\n\nAssistant:"

        # Claude 호출
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": chat_messages
                }
            ]
        )

        answer = response.content[0].text
        st.session_state.chat_history.append((user_input, answer))

# 채팅 출력
for q, a in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
