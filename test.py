# streamlit_rag_manual_chatbot.py

import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatAnthropic
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# 환경 변수 로드
load_dotenv()
claude_api_key = os.getenv("ANTHROPIC_API_KEY")

# Tesseract 설치 경로 설정 (윈도우)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

st.set_page_config(page_title="📱 전자제품 설명서 Claude 챗봇")
st.title("📱 전자제품 설명서 기반 Claude RAG 챗봇")
st.markdown("전자제품 PDF 설명서를 업로드하고, 궁금한 점을 물어보세요. 예: *배터리 교체 방법은?*")

# 세션 상태 초기화
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_uploaded_file_name" not in st.session_state:
    st.session_state.last_uploaded_file_name = None

# PDF 업로드
uploaded_file = st.file_uploader("전자제품 설명서 PDF 업로드", type="pdf")

# OCR 텍스트 추출 함수
def extract_text_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text += pytesseract.image_to_string(img, lang="kor+eng") + "\n"
    return text

# 벡터스토어 구축 함수
def build_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    documents = [Document(page_content=t) for t in texts]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# 벡터스토어 생성 (업로드된 파일이 새 파일일 때만 처리)
if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_file_name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    with st.spinner("OCR 및 임베딩 처리 중..."):
        extracted_text = extract_text_with_ocr(tmp_path)
        vs = build_vectorstore(extracted_text)
        st.session_state.vectorstore = vs
        st.session_state.last_uploaded_file_name = uploaded_file.name
        st.success("PDF 문서 처리 완료! 이제 질문해보세요.")

# 질문 처리
query = st.chat_input("전자제품 설명서에서 궁금한 점을 물어보세요")

if query and st.session_state.vectorstore:
    with st.spinner("Claude가 설명서를 기반으로 답변 중..."):
        retriever = st.session_state.vectorstore.as_retriever()
        llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0, anthropic_api_key=claude_api_key)
        chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        docs = retriever.get_relevant_documents(query)
        result = chain({"input_documents": docs, "question": "한국어로 답변해줘: " + query}, return_only_outputs=True)
        st.chat_message("user").markdown(query)
        st.chat_message("assistant").markdown(result["output_text"])
elif query:
    st.warning("먼저 설명서 PDF를 업로드해주세요. (예: '갤럭시 울트라24 울트라.pdf')")
