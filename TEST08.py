# 좋은 버전
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
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def get_bert_model():
    return SentenceTransformer("jhgan/ko-sbert-sts")

def get_bert_similarity(answer1, answer2):
    model = get_bert_model()
    emb1 = model.encode(answer1, convert_to_tensor=True)
    emb2 = model.encode(answer2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return round(similarity * 100, 2)  # 퍼센트로 반환



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
menu = st.sidebar.radio("📂 메뉴 선택", ["📎 파일 업로드", "💬 챗봇", "📊 응답 비교"])

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

            아래 분류 리스트 중 하나를 반드시 선택하여 답해주세요. 다른 단어나 설명 없이 정확히 분류 이름만 한 줄로 출력하세요.


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


    st.subheader("📦 제품 카테고리를 선택해주세요")
    icons = {
        "모바일": "📱", "TV/영상.음향": "📺", "가전": "🧺", "PC / 프린터": "💻",
        "메모리 & 스토리지": "💾", "디스플레이": "🖥️", "카메라 & 캠코더": "📷"
    }
    cols = st.columns(4)
    for i, category in enumerate(icons.keys()):
        with cols[i % 4]:
            if category in categories:
                if st.button(f"{icons[category]} {category}", key=category):
                    st.session_state.selected_category = category
            else:
                st.button(f"{icons[category]} {category}", key=category, disabled=True)
    # 선택된 카테고리 값
    selected_category = st.session_state.get("selected_category", None)

    #제품명선택
    if selected_category:
        products = [d["product_name"] for d in st.session_state.uploaded_data if d["category"] == selected_category]
        
        if products:
            # 👉 기본 안내 문구 추가
            product_options = ["제품명을 고르세요"] + products
            selected_product = st.selectbox("🏷️ 제품명 선택", product_options)

            # 실제 제품이 선택된 경우에만 진행
            if selected_product != "제품명을 고르세요":
                product_data = next((d for d in st.session_state.uploaded_data if d["product_name"] == selected_product), None)
                if product_data:
                    product_text = product_data["text"]
                    
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = splitter.create_documents([product_text])
                    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

                    def format_chat(chat_pairs):
                        return "\n".join([f"{'사용자' if r == 'user' else '챗봇'}: {m}" for r, m in chat_pairs[:-1]])
                else:
                    st.warning("❗ 선택한 제품 정보를 찾을 수 없습니다.")
        else:
            st.info("📭 해당 카테고리에 제품이 없습니다.")
    else:
        st.info("📦 먼저 제품 카테고리를 선택해주세요.")


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

        # 🔽 여기에 추가
        st.session_state.last_question = user_input
        st.session_state.last_rag_answer = answer

        base_4o_prompt = PromptTemplate.from_template("""[질문]\n{question}\n\n[답변]""")
        base_4o_chain = (
            {"question": lambda _: user_input}
            | base_4o_prompt
            | ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
            | StrOutputParser()
        )
        st.session_state.last_4o_answer = base_4o_chain.invoke("dummy")

        base_35_chain = (
            {"question": lambda _: user_input}
            | base_4o_prompt
            | ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
            | StrOutputParser()
        )
        st.session_state.last_35_answer = base_35_chain.invoke("dummy")

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    if 'product_text' in locals():
        with st.expander("📄 청크 미리보기"):
            preview_chunks = splitter.split_text(product_text)
            for i, chunk in enumerate(preview_chunks):
                st.markdown(f"**Chunk {i+1}**")
                st.code(chunk)


elif menu == "📊 응답 비교":
    st.title("📊 RAG vs GPT-4o vs GPT-3.5 응답 비교")

    if not all(k in st.session_state for k in ["last_question", "last_rag_answer", "last_4o_answer", "last_35_answer"]):
        st.info("💬 챗봇에서 먼저 질문하고 응답을 생성하세요.")
        st.stop()

    st.subheader("❓ 질문")
    st.markdown(f"> {st.session_state.last_question}")

    st.subheader("🧠 RAG 기반 응답")
    st.markdown(st.session_state.last_rag_answer)

    st.subheader("🧠 GPT-4o (RAG 없이)")
    st.markdown(st.session_state.last_4o_answer)

    st.subheader("🧠 GPT-3.5 (RAG 없이)")
    st.markdown(st.session_state.last_35_answer)

    st.divider()

    st.subheader("📐 BERT 기반 유사도 (%)")
    col1, col2, col3 = st.columns(3)

    with col1:
        score_4o = get_bert_similarity(st.session_state.last_rag_answer, st.session_state.last_4o_answer)
        st.metric("RAG vs GPT-4o", f"{score_4o}%")
    with col2:
        score_35 = get_bert_similarity(st.session_state.last_rag_answer, st.session_state.last_35_answer)
        st.metric("RAG vs GPT-3.5", f"{score_35}%")
    with col3:
        score_4o_35 = get_bert_similarity(st.session_state.last_4o_answer, st.session_state.last_35_answer)
        st.metric("GPT-4o vs GPT-3.5", f"{score_4o_35}%")


