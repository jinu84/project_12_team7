import os
import fitz  # PyMuPDF
import tempfile
import streamlit as st
import hashlib
import re
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain.schema import Document

# ======================= 전처리 함수 =======================

def remove_unusual_unicode(text: str) -> str:
    return re.sub(r"[^\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F"
                  r"\u0041-\u005A\u0061-\u007A"
                  r"\u0030-\u0039"
                  r"\u0020-\u007E"
                  r"\u3000-\u303F\uFF00-\uFFEF"
                  r"\u2010-\u205E"
                  r"\n\r\t .,;:?!\"'()\-\u2013\u2014_=+/\\[\]{}<>%@&#~"
                  r"]+", "", text)

def clean_newlines(text: str) -> str:
    lines = text.split('\n')
    cleaned = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if i == len(lines) - 1:
            cleaned.append(line)
            break
        if not re.search(r"[.!?…]$|습니다$|합니다$|되며$|됩니다$", line):
            cleaned.append(line + ' ')
        else:
            cleaned.append(line + '\n')
    return ''.join(cleaned)

def extract_text_without_images(doc):
    cleaned_texts = []
    for page in doc:
        blocks = page.get_text("blocks")
        text_blocks = [block[4] for block in blocks if block[6] == 0]
        page_text = "\n".join(text_blocks).strip()
        page_text = remove_unusual_unicode(page_text)
        cleaned_texts.append(page_text)
    return cleaned_texts

def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# ======================= 환경 변수 및 초기 설정 =======================

load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

st.set_page_config(page_title="📚 PDF 선택 기반 질문응답", page_icon="📄")
st.title("📘 선택한 PDF 기반 질문응답 (RAG vs LLM)")

uploaded_pdfs = st.file_uploader("📎 여러 PDF 파일을 업로드하세요", type=["pdf"], accept_multiple_files=True)

# ======================= 처리 로직 =======================

if uploaded_pdfs:
    pdf_name_to_vectorstore = {}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

    os.makedirs("faiss_index", exist_ok=True)

    for uploaded_pdf in uploaded_pdfs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_path = tmp_file.name

        file_hash = calculate_file_hash(tmp_path)
        index_path = os.path.join("faiss_index", file_hash)

        if os.path.exists(f"{index_path}/index.faiss") and os.path.exists(f"{index_path}/index.pkl"):
            vectorstore = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        else:
            doc = fitz.open(tmp_path)
            page_texts = extract_text_without_images(doc)
            doc.close()

            cleaned_text = clean_newlines("\n".join(page_texts))
            docs = text_splitter.create_documents([cleaned_text])

            # ✅ ID 중복 방지를 위해 새 Document로 생성
            docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in docs]

            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(index_path)

        pdf_name_to_vectorstore[uploaded_pdf.name] = vectorstore

    # ======================= 질문 섹션 =======================

    selected_pdf_name = st.selectbox("📄 질문할 PDF를 선택하세요", list(pdf_name_to_vectorstore.keys()))
    question = st.text_input("❓ 질문을 입력하세요")

    if question and selected_pdf_name:
        selected_vectorstore = pdf_name_to_vectorstore[selected_pdf_name]
        retriever = selected_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})

        # RAG 프롬프트
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

        with st.spinner("🤖 RAG 기반 응답 생성 중..."):
            rag_response = rag_chain.invoke(question)

        with st.spinner("🧠 LLM 단독 응답 생성 중..."):
            direct_prompt = f"""
            너는 사용자의 질문에 문서 없이도 정확하게 답하는 한국어 챗봇이야.
            질문에 대해 알고 있는 지식으로 최대한 성실하게 답해줘.

            [질문]
            {question}

            [답변]
            """
            llm_response = llm.invoke(direct_prompt)

        st.markdown(f"### 🔎 RAG 응답 (📄 {selected_pdf_name})")
        st.success(rag_response)

        st.markdown("### 🧠 LLM 단독 응답")
        st.info(llm_response)
