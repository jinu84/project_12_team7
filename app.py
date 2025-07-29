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

# Streamlit UI
st.set_page_config(page_title="📚 PDF RAG 응답", page_icon="📄")
st.title("📘 PDF 기반 질문응답 (RAG 기반)")

uploaded_pdf = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
question = st.text_input("❓ 질문을 입력하세요")

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    doc = fitz.open(tmp_path)
    page_texts = [page.get_text() for page in doc]
    doc.close()

    # 전체 텍스트 확인 (선택적으로 사용 가능)
    with st.expander("📝 전체 파싱된 PDF 텍스트 (페이지별)"):
        for idx, text in enumerate(page_texts):
            with st.expander(f"📄 Page {idx + 1}"):
                st.text(text)

    # LangChain 기반 처리
    with st.spinner("🔍 PDF 청크 생성 중..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", " "]
        )
        all_texts = "\n".join(page_texts)
        docs = text_splitter.create_documents([all_texts])

        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})

        prompt = PromptTemplate.from_template("""
        너는 업로드된 PDF 문서에 기반해서 질문에 정확하게 답하는 한국어 챗봇이야.
        아래 문맥을 참고해서 질문에 답하고, 문맥에 없으면 '문서에 해당 정보를 찾을 수 없습니다.'라고 답해.

        [문맥]
        {context}

        [질문]
        {question}

        [답변]
        """)

        llm = ChatAnthropic(model="claude-3-haiku-20240307", api_key=CLAUDE_API_KEY, temperature=0)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

# 질문 입력 시 RAG 응답만 출력
if question and uploaded_pdf:
    with st.spinner("🤖 RAG 응답 생성 중..."):
        rag_response = rag_chain.invoke(question)

    st.markdown("### 🔎 RAG 기반 응답")
    st.success(f"**Claude + PDF Context:**\n\n{rag_response}")
