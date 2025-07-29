import os
import fitz  # PyMuPDF
import tempfile
import streamlit as st
from dotenv import load_dotenv
import re
import unicodedata

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic

# 🔧 줄바꿈 전처리 함수
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

# ✅ 보기 드문 유니코드 문자 제거 필터
def remove_unusual_unicode(text: str) -> str:
    return re.sub(r"[^\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F"  # 한글
                  r"\u0041-\u005A\u0061-\u007A"                 # 영어
                  r"\u0030-\u0039"                              # 숫자
                  r"\u0020-\u007E"                              # 일반 특수기호
                  r"\u3000-\u303F\uFF00-\uFFEF"                 # 전각 기호
                  r"\u2010-\u205E"                              # dash, quote 등
                  r"\n\r\t .,;:?!\"'()\-–—_=+/\\[\]{}<>%@&#~"    # 일반 기호
                  r"]+", "", text)

# 📤 이미지 제거: 텍스트 블록만 추출
def extract_text_without_images(doc):
    cleaned_texts = []
    for page in doc:
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, type)
        text_blocks = [block[4] for block in blocks if block[6] == 0]  # block type 0 = text only
        page_text = "\n".join(text_blocks).strip()
        page_text = remove_unusual_unicode(page_text)  # 🔹 이상한 문자 제거 적용
        cleaned_texts.append(page_text)
    return cleaned_texts

# 🌱 환경 변수 로드
load_dotenv()
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# 🎛️ Streamlit UI
st.set_page_config(page_title="📚 PDF RAG vs LLM 비교", page_icon="📄")
st.title("📘 PDF 기반 질문응답 (RAG vs LLM 비교)")

# 📂 PDF 업로드
uploaded_pdf = st.file_uploader("📎 PDF 파일을 업로드하세요", type=["pdf"])
question = st.text_input("❓ 질문을 입력하세요")

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    doc = fitz.open(tmp_path)
    page_texts = extract_text_without_images(doc)
    doc.close()

    # 📝 페이지별 원문 확인
    with st.expander("📝 전체 파싱된 PDF 텍스트 (페이지별)"):
        for idx, text in enumerate(page_texts):
            with st.expander(f"📄 Page {idx + 1}"):
                st.text(text)

    # 📌 전처리 + 청크 생성
    with st.spinner("🔍 PDF 청크 생성 중..."):
        merged_text = "\n".join(page_texts)
        cleaned_text = clean_newlines(merged_text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", ".", " ", "\n"]
        )
        docs = text_splitter.create_documents([cleaned_text])

        with st.expander("🧩 생성된 청크 확인"):
            for idx, doc in enumerate(docs):
                with st.expander(f"🔹 청크 {idx + 1} (길이: {len(doc.page_content)}자)"):
                    st.text(doc.page_content)

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

    if question:
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

        st.markdown("### 🔎 RAG 기반 응답")
        st.success(f"**Claude + PDF Context:**\n\n{rag_response}")

        st.markdown("### 🧠 LLM 단독 응답")
        st.info(f"**Claude 단독 응답:**\n\n{llm_response}")
