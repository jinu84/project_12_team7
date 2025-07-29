import streamlit as st  # ✅ 반드시 필요

# 📂 사이드바 메뉴 구성
menu = st.sidebar.radio("📂 메뉴 선택", ["💬 Chatbot", "📊 유사도 비교"])

if menu == "💬 Chatbot":
    # 여기 안에 Chatbot 코드 전체 복사해서 넣기
    # 지금까지 작성하신 모든 코드 그대로!
    
    import os
    import fitz  # PyMuPDF
    import tempfile

    from dotenv import load_dotenv
    import re
    import hashlib

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import Document


    # 🔧 전처리 함수
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

    def remove_unusual_unicode(text: str) -> str:
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

    def calculate_file_hash(file_path):
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # 🌱 환경변수 로드
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # 🌐 Streamlit 페이지 설정
    st.set_page_config(page_title="📚 PDF 기반 챗봇 (Claude + RAG)", page_icon="📄")
    st.title("📘 PDF 기반 챗봇 (Claude + RAG)")

    # ✅ 제품 분류 정의
    product_categories = {
        "모바일": ["스마트폰", "휴대폰"],
        "TV/영상.음향": ["텔레비전", "TV", "사운드바"],
        "가전": ["냉장고", "세탁기", "에어드레서", "청소기", "건조기"],
        "PC / 프린터": ["컴퓨터", "노트북", "프린터", "복합기"],
        "메모리 & 스토리지": ["메모리", "SSD", "microSD", "스토리지"],
        "디스플레이": ["모니터", "디스플레이"],
        "카메라 & 캠코더": ["카메라", "캠코더"]
    }

    # 📎 여러 PDF 업로드
    uploaded_pdfs = st.file_uploader("📎 PDF 파일을 업로드하세요", type=["pdf"], accept_multiple_files=True)

    highlighted = set()
    category_to_products = {}
    product_to_text = {}
    all_page_texts = []

    # 상태 초기화
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
    if "selected_product" not in st.session_state:
        st.session_state.selected_product = None

    # 📄 문서 처리 및 카테고리/제품명 추출
    if uploaded_pdfs:
        for uploaded_pdf in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                tmp_path = tmp_file.name

            doc = fitz.open(tmp_path)
            page_texts = extract_text_without_images(doc)
            doc.close()

            merged_text = "\n".join(page_texts)
            cleaned_text = clean_newlines(merged_text)
            all_page_texts.extend(page_texts)

            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            # 제품 카테고리 추론
            category_prompt = PromptTemplate.from_template("""
            아래는 제품 분류 리스트입니다:
            {categories}

            위 분류 중 업로드된 문서 내용에 가장 관련 있는 제품 분류 하나만 정확히 골라주세요.

            [문서 내용 요약]
            {text}

            [관련 제품 분류]""")
            category_chain = (
                {"categories": lambda x: ", ".join(product_categories.keys()), "text": lambda x: cleaned_text[:1500]}
                | category_prompt
                | llm
                | StrOutputParser()
            )
            category_response = category_chain.invoke("dummy")

            matched_category = None
            for label in product_categories:
                if label in category_response:
                    highlighted.add(label)
                    matched_category = label
                    break

            # 제품명 추론
            name_prompt = PromptTemplate.from_template("""
            아래는 전자제품 설명서의 본문 내용 일부입니다. 이 내용을 읽고 제품명을 정확히 한 줄로 알려주세요. 예: 갤럭시 S24 울트라, 삼성 비스포크 세탁기 등

            [설명서 내용]
            {text}

            [제품명]""")
            name_chain = (
                {"text": lambda x: cleaned_text[:2000]}
                | name_prompt
                | llm
                | StrOutputParser()
            )
            name_response = name_chain.invoke("dummy").strip()

            if matched_category:
                if name_response not in category_to_products.get(matched_category, []):
                    category_to_products.setdefault(matched_category, []).append(name_response)
                    product_to_text[name_response] = cleaned_text

    # 📦 카테고리 버튼
    st.subheader("📦 제품 카테고리를 선택해주세요")
    items = [
        ("📱", "모바일"), ("📺", "TV/영상.음향"),
        ("🧺", "가전"), ("💻", "PC / 프린터"),
        ("💾", "메모리 & 스토리지"), ("🖥️", "디스플레이"),
        ("📷", "카메라 & 캠코더")
    ]
    rows = [items[i:i+4] for i in range(0, len(items), 4)]
    for row in rows:
        cols = st.columns(len(row), gap="small")
        for col, (icon, label) in zip(cols, row):
            with col:
                if label in highlighted:
                    if st.button(f"{icon}\n{label}", key=label, use_container_width=True):
                        st.session_state.selected_category = label
                else:
                    st.button(f"{icon}\n{label}", key=label, use_container_width=True, disabled=True)

    # 🏷️ 제품명 선택
    selected_category = st.session_state.selected_category
    if selected_category and selected_category in category_to_products:
        st.subheader("🏷️ 해당 카테고리의 제품명")
        selected_product = st.selectbox(
            "제품명을 선택하세요",
            category_to_products[selected_category],
            index=category_to_products[selected_category].index(st.session_state.selected_product)
            if st.session_state.selected_product in category_to_products[selected_category]
            else 0,
            key="selected_product"  # 주의: key 설정 후 외부에서 이 값을 set하지 말 것
        )


    # 💬 챗봇 영역
    if st.session_state.get("selected_product") not in [None, ""]:
        selected_product = st.session_state.selected_product
        product_text = product_to_text[selected_product]

        # 벡터스토어 구성
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([product_text])
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        # 대화 히스토리 포맷 함수
        def format_chat_history(chat_pairs):
            return "\n".join([f"{'사용자' if role == 'user' else '챗봇'}: {msg}" for role, msg in chat_pairs[:-1]])

        # 프롬프트 템플릿
        prompt = PromptTemplate.from_template("""
        너는 업로드된 PDF 문서를 기반으로 질문에 답변하는 한국어 챗봇이야.
        아래 대화 내용을 참고하여, 가장 마지막 질문에 답해줘.
        문서에 관련 정보가 없으면 "문서에 해당 정보를 찾을 수 없습니다."라고 답변해줘.

        [대화 히스토리]
        {chat_history}

        [질문]
        {question}

        [문맥]
        {context}

        [답변]
        """)

        # LLM 초기화
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # RAG 체인
        rag_chain = (
            {
                "question": lambda h: h[-1][1],
                "chat_history": lambda h: format_chat_history(h),
                "context": lambda h: retriever.invoke(h[-1][1])
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # 세션 상태 초기화
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # 사용자 입력 처리
        if question := st.chat_input("💬 궁금한 점을 입력하세요!"):
            st.session_state.chat_history.append(("user", question))
            with st.spinner("🤖 RAG 응답 생성 중..."):
                answer = rag_chain.invoke(st.session_state.chat_history)
            st.session_state.chat_history.append(("ai", answer))

        # 채팅 표시
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(
                    f"<div style='background-color: {'#DCF8C6' if role == 'user' else '#E6ECF0'}; "
                    f"padding: 10px; border-radius: 8px'>{msg}</div>", unsafe_allow_html=True
                )

            # ✅ 기본 GPT-4o 응답 보기
        if "chat_history" in st.session_state and len(st.session_state.chat_history) >= 2:
            last_user_msg = st.session_state.chat_history[-2][1]  # 직전 사용자 질문
            with st.expander("🤖 기본 GPT-4o 응답 보기"):
                base_llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )

                base_prompt = PromptTemplate.from_template("""
                다음 사용자 질문에 일반적인 GPT-4o 모델로 답변하세요.
                문서 기반이 아닌 일반 지식에 기반해서 자연스럽게 답변해 주세요.

                [질문]
                {question}

                [답변]
                """)

                base_chain = (
                    {"question": lambda _: last_user_msg}
                    | base_prompt
                    | base_llm
                    | StrOutputParser()
                )

                base_answer = base_chain.invoke("dummy")  # 입력값은 무시됨
                st.markdown(base_answer)

                # ✅ 기본 GPT-3.5 응답 보기
        if "chat_history" in st.session_state and len(st.session_state.chat_history) >= 2:
            last_user_msg = st.session_state.chat_history[-2][1]  # 직전 사용자 질문
            with st.expander("🤖 기본 GPT-3.5 응답 보기"):
                gpt35_llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )

                gpt35_prompt = PromptTemplate.from_template("""
                다음 사용자 질문에 일반적인 GPT-3.5 모델로 답변하세요.
                문서 기반이 아닌 일반 지식에 기반해서 자연스럽게 답변해 주세요.

                [질문]
                {question}

                [답변]
                """)

                gpt35_chain = (
                    {"question": lambda _: last_user_msg}
                    | gpt35_prompt
                    | gpt35_llm
                    | StrOutputParser()
                )

                gpt35_answer = gpt35_chain.invoke("dummy")  # 입력값은 무시됨
                st.markdown(gpt35_answer)


        # ✅ 청크 미리보기는 여기! 루프 밖에 있어야 항상 보입니다
        with st.expander("📄 청크 미리보기"):
            preview_chunks = text_splitter.split_text(product_text)
            for i, chunk in enumerate(preview_chunks):
                st.markdown(f"**Chunk {i+1}**")
                st.code(chunk)

                from sentence_transformers import SentenceTransformer, util

if menu == "📊 유사도 비교":
    st.subheader("📊 응답 유사도 비교 (BERT 기반)")

    # 응답 가져오기 (이전 코드에서 저장된 내용)
    rag_answer = st.session_state.chat_history[-1][1] if st.session_state.chat_history else ""
    gpt4o_answer = st.session_state.get("gpt4o_base_answer", "")
    gpt35_answer = st.session_state.get("gpt35_base_answer", "")

    if rag_answer and gpt4o_answer and gpt35_answer:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        rag_emb = model.encode(rag_answer, convert_to_tensor=True)
        gpt4o_emb = model.encode(gpt4o_answer, convert_to_tensor=True)
        gpt35_emb = model.encode(gpt35_answer, convert_to_tensor=True)

        sim_rag_gpt4o = util.cos_sim(rag_emb, gpt4o_emb).item()
        sim_rag_gpt35 = util.cos_sim(rag_emb, gpt35_emb).item()
        sim_gpt4o_gpt35 = util.cos_sim(gpt4o_emb, gpt35_emb).item()

        st.markdown("#### ✅ 응답 간 유사도 (Cosine Similarity)")
        st.write(f"📘 **RAG vs GPT-4o**: {sim_rag_gpt4o:.4f}")
        st.write(f"📘 **RAG vs GPT-3.5**: {sim_rag_gpt35:.4f}")
        st.write(f"📘 **GPT-4o vs GPT-3.5**: {sim_gpt4o_gpt35:.4f}")
    else:
        st.warning("RAG 또는 기본 GPT 응답이 존재하지 않습니다. 먼저 챗봇에서 질문을 입력해주세요.")


