import streamlit as st
import os
import tempfile
from document_processor import load_document, split_documents, create_vector_db, load_vector_db, load_documents_from_folder
from rag_chain import create_rag_chain, ask_question

st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="📚",
    layout="wide"
)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

if 'document_count' not in st.session_state:
    st.session_state.document_count = 0

if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0

def init_rag_system():
    try:
        st.session_state.qa_chain = create_rag_chain()
        st.session_state.db_initialized = True
        return True
    except Exception as e:
        st.error(f"初始化RAG系统失败: {e}")
        return False

def process_uploaded_files(uploaded_files):
    if not uploaded_files:
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        documents = []
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            text = load_document(file_path)
            if text.strip():
                documents.append(text)
        
        if not documents:
            st.warning("未找到有效文档内容")
            return
        
        chunks = split_documents(documents)
        st.session_state.chunk_count += len(chunks)
        st.session_state.document_count += len(documents)
        
        try:
            create_vector_db(chunks)
            st.success(f"成功处理 {len(documents)} 个文档，生成 {len(chunks)} 个文本块")
            
            init_rag_system()
        except Exception as e:
            st.error(f"创建向量数据库失败: {e}")

def main():
    st.title("📚 RAG智能问答系统")
    st.markdown("---")
    
    with st.sidebar:
        st.header("知识库管理")
        
        uploaded_files = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="支持PDF、DOCX、TXT格式"
        )
        
        if st.button("构建/更新知识库", key="build_kb"):
            if uploaded_files:
                with st.spinner("正在处理文档..."):
                    process_uploaded_files(uploaded_files)
            else:
                st.warning("请先上传文档")
        
        st.markdown("---")
        st.subheader("知识库状态")
        st.info(f"已加载文档数: {st.session_state.document_count}")
        st.info(f"文本块总数: {st.session_state.chunk_count}")
        
        if st.button("初始化RAG系统"):
            with st.spinner("正在初始化..."):
                if init_rag_system():
                    st.success("RAG系统初始化成功!")
        
        if st.button("清除对话历史"):
            st.session_state.chat_history = []
            st.session_state.qa_chain = None
            st.session_state.db_initialized = False
            st.success("对话历史已清除")
    
    st.header("问答交互")
    
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.expander(f"Q: {question}", expanded=False):
            st.markdown(f"**A:** {answer}")
    
    question = st.text_input("请输入您的问题:", key="question_input")
    
    if st.button("提问", key="ask_button"):
        if not question.strip():
            st.warning("请输入问题")
        elif not st.session_state.db_initialized:
            st.warning("请先构建知识库或初始化RAG系统")
        else:
            with st.spinner("正在思考..."):
                result = ask_question(st.session_state.qa_chain, question, st.session_state.chat_history)
                answer = result["answer"]
                
                st.session_state.chat_history.append((question, answer))
                
                st.markdown("---")
                st.subheader("回答")
                st.write(answer)
                
                if result.get("source_documents"):
                    with st.expander("查看参考文档"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.markdown(f"**文档 {i}:**")
                            st.write(doc)
    
    st.markdown("---")
    st.caption("基于Ollama + LangChain + Streamlit构建的RAG智能问答系统")

if __name__ == "__main__":
    main()