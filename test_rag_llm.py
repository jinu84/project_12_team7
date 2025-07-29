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

import pytesseract
from PIL import Image
import io

# 🔧 OCR 경로 설정 (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🧠 PDF 페이지별 텍스트 추출 (OCR fallback 포함)
def extract_text_with_fallback(doc):
    page_texts = []
    for page in doc:
        text = page.get_text()
        if len(text.strip()) < 30 or "�" in text:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="kor+eng")
        page_texts.append(text)
    return page_texts

# 🔐 환경 변수 로드
load_dotenv()
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# 🎛️ Streamlit UI 설정
st.set_page_config(page_title="📚 PDF RAG vs LLM 비교", page_icon="📄")
st.title("📘 PDF 기반 질문응답 (RAG vs LLM 비교)")

# 📁 파일 업로드 + 질의 입력
uploaded_pdf = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
question = st.text_input("❓ 질문을 입력하세요")

# 🧾 파일 업로드 처리
if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    doc = fitz.open(tmp_path)
    page_texts = extract_text_with_fallback(doc)
    doc.close()

    # ✂️ 텍스트 청크 분할
    with st.spinner("🔍 PDF 청크 생성 중..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", "•", "▶", ":", "1.", "2.", "3.", "-", ". "]
        )
        all_texts = "\n".join(page_texts)
        docs = text_splitter.create_documents([all_texts])

        # ✅ 청크 미리보기만 표시
        with st.expander("📑 분할된 청크 미리보기"):
            for idx, doc in enumerate(docs):
                st.markdown(f"**청크 {idx + 1}**\n\n{doc.page_content}")

        # 🔍 임베딩 및 벡터스토어 생성
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})

        # 🧠 프롬프트 구성
        prompt = PromptTemplate.from_template("""
        너는 업로드된 PDF 문서에 기반해서 질문에 정확하게 답하는 한국어 챗봇이야.
        아래 문맥을 참고해서 질문에 답하고, 문맥에 없으면 '문서에 해당 정보를 찾을 수 없습니다.'라고 답해.

        [문맥]
        {context}

        [질문]
        {question}

        [답변]
        """)

        # 🤖 LLM 호출 설정 (Claude)
        llm = ChatAnthropic(model="claude-3-haiku-20240307", api_key=CLAUDE_API_KEY, temperature=0)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

# ❓ 질문 입력 시 응답 생성
if question and uploaded_pdf:
    with st.spinner("🤖 RAG 응답 생성 중..."):
        rag_response = rag_chain.invoke(question)

    with st.spinner("🤖 LLM 단독 응답 생성 중..."):
        direct_prompt = f"""
        너는 사용자의 질문에 문서 없이도 정확하게 답하는 한국어 챗봇이야.
        질문에 대해 알고 있는 지식으로 최대한 성실하게 답해줘.

        [질문]
        {question}

        [답변]
        """
        llm_response = llm.invoke(direct_prompt)

    st.markdown("### 🔎 RAG 기반 응답")
    st.success(f"**Claude + PDF Context:**\n\n{rag_response}")

    st.markdown("### 🧠 LLM 단독 응답")
    st.info(f"**Claude 단독 응답:**\n\n{llm_response}")
