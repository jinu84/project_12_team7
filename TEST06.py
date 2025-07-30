import os
import re
import fitz  # PyMuPDF
import tempfile
import hashlib
import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI


# 🔐 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 📚 제품 카테고리
product_categories = {
    "모바일": ["스마트폰", "휴대폰"],
    "TV/영상.음향": ["텔레비전", "TV", "사운드바"],
    "가전": ["냉장고", "세탁기", "에어드레서", "청소기", "건조기"],
    "PC / 프린터": ["컴퓨터", "노트북", "프린터", "복합기"],
    "메모리 & 스토리지": ["메모리", "SSD", "microSD", "스토리지"],
    "디스플레이": ["모니터", "디스플레이"],
    "카메라 & 캠코더": ["카메라", "캠코더"]
}

# 🔧 텍스트 전처리 함수
def clean_newlines(text):
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

def remove_unusual_unicode(text):
    return re.sub(r"[^\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F"
                  r"\u0041-\u005A\u0061-\u007A"
                  r"\u0030-\u0039"
                  r"\u0020-\u007E"
                  r"\u3000-\u303F\uFF00-\uFFEF"
                  r"\u2010-\u205E"
                  r"\n\r\t .,;:?!\"'()\-\u2013\u2014_=+/\\[\]{}<>%@&#~"
                  r"]+", "", text)

def extract_text_without_images(doc):
    cleaned_texts = []
    for page in doc:
        blocks = page.get_text("blocks")
        text_blocks = [block[4] for block in blocks if block[6] == 0]
        page_text = "\n".join(text_blocks).strip()
        page_text = remove_unusual_unicode(page_text)
        cleaned_texts.append(page_text)
    return cleaned_texts


# 🌐 Streamlit 설정
st.set_page_config(page_title="📚 PDF 기반 챗봇", page_icon="📄")
menu = st.sidebar.radio("📂 메뉴 선택", ["📎 파일 업로드", "💬 챗봇"])

# 🔄 공유 데이터 초기화
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📎 파일 업로드 화면
if menu == "📎 파일 업로드":
    st.title("📎 PDF 업로드 및 제품 분류")

    uploaded_pdfs = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"], accept_multiple_files=True)

    if uploaded_pdfs:
        st.session_state.uploaded_data.clear()
        for uploaded_pdf in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                tmp_path = tmp_file.name

            doc = fitz.open(tmp_path)
            page_texts = extract_text_without_images(doc)
            doc.close()

            full_text = "\n".join(page_texts)
            cleaned_text = clean_newlines(full_text)

            llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

            category_prompt = PromptTemplate.from_template("""
            아래는 제품 분류 리스트입니다:
            {categories}

            위 분류 중 업로드된 문서 내용에 가장 관련 있는 제품 분류 하나만 골라주세요.

            [문서 내용 요약]
            {text}

            [관련 제품 분류]""")
            category_chain = (
                {"categories": lambda _: ", ".join(product_categories.keys()), "text": lambda _: cleaned_text[:1500]}
                | category_prompt
                | llm
                | StrOutputParser()
            )
            category = category_chain.invoke("dummy")

            matched_category = next((c for c in product_categories if c in category), None)

            name_prompt = PromptTemplate.from_template("""
            아래 설명서 내용을 보고 제품명을 정확히 한 줄로 알려주세요. 예: 갤럭시 S24 울트라

            [설명서 내용]
            {text}

            [제품명]""")
            name_chain = (
                {"text": lambda _: cleaned_text[:2000]}
                | name_prompt
                | llm
                | StrOutputParser()
            )
            name = name_chain.invoke("dummy").strip()

            if matched_category:
                st.session_state.uploaded_data.append({
                    "category": matched_category,
                    "product_name": name,
                    "text": cleaned_text
                })

        st.success("✅ 문서 분류 및 제품명 추출이 완료되었습니다! 챗봇 탭으로 이동해 질문해보세요.")

# 💬 챗봇 화면
elif menu == "💬 챗봇":
    st.title("💬 PDF 기반 RAG 챗봇")

    if not st.session_state.uploaded_data:
        st.warning("📎 먼저 파일을 업로드하고 제품을 등록해주세요.")
        st.stop()

    categories = list(set(d["category"] for d in st.session_state.uploaded_data))
    selected_category = st.selectbox("📦 카테고리 선택", categories)

    products = [d["product_name"] for d in st.session_state.uploaded_data if d["category"] == selected_category]
    selected_product = st.selectbox("🏷️ 제품명 선택", products)

    product_data = next((d for d in st.session_state.uploaded_data if d["product_name"] == selected_product), None)
    product_text = product_data["text"]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([product_text])
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    def format_chat(chat_pairs):
        return "\n".join([f"{'사용자' if r == 'user' else '챗봇'}: {m}" for r, m in chat_pairs[:-1]])

    prompt = PromptTemplate.from_template("""
    너는 업로드된 PDF 문서를 기반으로 질문에 답변하는 한국어 챗봇이야.
    문서에 없는 질문이면 "문서에 해당 정보를 찾을 수 없습니다."라고 말해줘.

    [대화 히스토리]
    {chat_history}

    [질문]
    {question}

    [문맥]
    {context}

    [답변]
    """)

    rag_chain = (
        {
            "question": lambda h: h[-1][1],
            "chat_history": lambda h: format_chat(h),
            "context": lambda h: retriever.invoke(h[-1][1])
        }
        | prompt
        | ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
        | StrOutputParser()
    )

    if user_input := st.chat_input("💬 질문을 입력하세요!"):
        st.session_state.chat_history.append(("user", user_input))
        with st.spinner("🤖 답변 생성 중..."):
            answer = rag_chain.invoke(st.session_state.chat_history)
        st.session_state.chat_history.append(("ai", answer))

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    with st.expander("📄 청크 미리보기"):
        for i, chunk in enumerate(splitter.split_text(product_text)):
            st.markdown(f"**Chunk {i+1}**")
            st.code(chunk)
