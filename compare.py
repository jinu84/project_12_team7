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

from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 🔐 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# 🎛️ Streamlit UI 설정
st.set_page_config(page_title="📚 PDF 기반 질문응답 비교", page_icon="📄")
st.title("📘 PDF 기반 RAG vs OpenAI vs Claude 비교 응답")

uploaded_pdf = st.file_uploader("📄 PDF 파일을 업로드하세요", type=["pdf"])
question = st.text_input("❓ 질문을 입력하세요")

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    doc = fitz.open(tmp_path)
    page_texts = [page.get_text() for page in doc]
    doc.close()

    # 🔍 텍스트 청크 생성
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " "]
    )
    all_texts = "\n".join(page_texts)
    docs = text_splitter.create_documents([all_texts])

    # 💡 임베딩 및 벡터스토어
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})

    # 🧠 공통 프롬프트
    prompt = PromptTemplate.from_template("""
    너는 업로드된 PDF 문서에 기반해서 질문에 정확하게 답하는 한국어 챗봇이야.
    아래 문맥을 참고해서 질문에 답하고, 문맥에 없으면 '문서에 해당 정보를 찾을 수 없습니다.'라고 답해.

    [문맥]
    {context}

    [질문]
    {question}

    [답변]
    """)

    # LLM 세팅
    llm_claude = ChatAnthropic(model="claude-3-haiku-20240307", api_key=CLAUDE_API_KEY, temperature=0)
    llm_openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0)

    rag_claude_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm_claude
        | StrOutputParser()
    )

    rag_openai_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm_openai
        | StrOutputParser()
    )

    # ✨ 응답 생성
    if question:
        with st.spinner("🤖 Claude + RAG 응답 생성 중..."):
            response_rag_claude = rag_claude_chain.invoke(question)

        with st.spinner("🤖 OpenAI + RAG 응답 생성 중..."):
            response_rag_openai = rag_openai_chain.invoke(question)

        with st.spinner("🧠 Claude 단독 응답 생성 중..."):
            direct_prompt = f"""
            너는 사용자의 질문에 문서 없이도 정확하게 답하는 한국어 챗봇이야.
            질문에 대해 알고 있는 지식으로 최대한 성실하게 답해줘.

            [질문]
            {question}

            [답변]
            """
            direct_response = llm_claude.invoke(direct_prompt)

        # 📝 출력
        st.subheader("🔎 응답 비교 결과")
        st.markdown("#### 📘 Claude + PDF RAG")
        st.success(response_rag_claude)

        st.markdown("#### 🤖 OpenAI + PDF RAG")
        st.info(response_rag_openai)

        st.markdown("#### 💬 Claude (문서 미포함)")
        st.warning(direct_response)