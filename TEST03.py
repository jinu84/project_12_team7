import os
import fitz  # PyMuPDF
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic

# 환경 변수 로드
load_dotenv()
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Streamlit UI 설정
st.set_page_config(page_title="📚 PDF RAG 응답", page_icon="📄", layout="wide")
st.title(":blue_book: PDF 기반 질문응답 (RAG 기반)")

if "pdf_required" not in st.session_state:
    st.session_state.pdf_required = False

st.subheader("📝 궁금하신 제품을 선택해주세요")

items = [
    ("🌀", "에어콘"),
    ("🧊", "냉장고/uae40치냉장고"),
    ("🧺", "세탁기/건조기/에어드레서"),
    ("📱", "모바일"),
    ("💻", "PC"),
    ("🖨️", "프린터/복합기"),
    ("📺", "TV"),
    ("🪟", "모니터"),
    ("🧹", "청소기"),
    ("🫧", "공기정정기/제습기"),
    ("🍳", "주방건전"),
    ("🔌", "스마트홈/기타")
]

# 스타일 추가
st.markdown("""
<style>
    div[data-testid="column"] > div {
        display: flex;
        justify-content: center;
    }
    button[kind="secondary"] {
        height: 90px !important;
        width: 90px !important;
        border-radius: 14px !important;
        font-size: 13px !important;
        white-space: pre-line;
        margin: 2px !important;
    }
</style>
""", unsafe_allow_html=True)

rows = [items[i:i+4] for i in range(0, len(items), 4)]
for row in rows:
    cols = st.columns([1, 1, 1, 1], gap="small")
    for col, (icon, label) in zip(cols, row):
        with col:
            if st.button(f"{icon}\n{label}", use_container_width=True):
                st.session_state.pdf_required = True
                st.rerun()

if st.session_state.pdf_required:
    st.warning("📄 선택하신 제품 관련 PDF를 업로드 해주세요")

    uploaded_pdf = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    if uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_path = tmp_file.name

        doc = fitz.open(tmp_path)
        page_texts = [page.get_text() for page in doc]
        doc.close()#dlkfa

        with st.expander("📝 전체 PDF 텍스트 (페이지별)"):
            for idx, text in enumerate(page_texts):
                with st.expander(f"📄 Page {idx + 1}"):
                    st.text(text)

        with st.spinner("🔍 PDF..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
            docs = splitter.create_documents(["\n".join(page_texts)])

            embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})

            prompt = PromptTemplate.from_template("""
            너는 업로드된 PDF 문서에 까지해서 질문에 정확하게 답해야 하는 한국어 책방 첫코야.
            문려를 보고 응답하라. 목록에 없으면 '\ubb38서에 해당 정보를 찾을 수 없습니다.'라고 답해.

            [문려]\n{context}
            [질문]\n{question}
            [답변]
            """)

            llm = ChatAnthropic(model="claude-3-haiku-20240307", api_key=CLAUDE_API_KEY, temperature=0)

            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if question := st.chat_input("💬 궁금한 점을 입력해주세요!"):
            st.session_state.chat_history.append(("user", question))
            with st.spinner("🤖 RAG 응답 생성 중..."):
                rag_response = rag_chain.invoke(question)
            st.session_state.chat_history.append(("ai", rag_response))

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)
