# -*- coding: utf-8 -*-
"""
RAG智能问答系统 - Streamlit Web应用
基于本地知识库的检索增强生成问答系统
"""

import streamlit as st
import os
import tempfile
from typing import List

import config
from document_loader import DocumentLoader
from vector_store import VectorStoreManager
from rag_chain import RAGQAChain

# 页面配置
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .doc-upload {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    .qa-section {
        background-color: #fafafa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# 初始化会话状态
def init_session_state():
    """初始化会话状态"""
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager()

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "docs_uploaded" not in st.session_state:
        st.session_state.docs_uploaded = False


def get_vector_store_stats() -> dict:
    """获取向量数据库统计信息"""
    try:
        return st.session_state.vector_store_manager.get_collection_stats()
    except Exception as e:
        return {"exists": False, "error": str(e)}


def process_uploaded_files(files: List) -> bool:
    """处理上传的文件"""
    if not files:
        return False

    loader = DocumentLoader()
    all_documents = []

    # 创建临时目录存储上传的文件
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            # 保存上传的文件到临时目录
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            try:
                # 加载并分割文档
                documents = loader.load_document(file_path)
                split_docs = loader.split_documents(documents)
                all_documents.extend(split_docs)
                st.success(f"✓ 已处理: {file.name}")
            except Exception as e:
                st.error(f"✗ 处理失败 {file.name}: {str(e)}")

    if all_documents:
        try:
            # 创建向量数据库
            st.session_state.vector_store_manager.create_vectorstore(all_documents)
            st.session_state.rag_chain = None  # 重置RAG链
            st.session_state.docs_uploaded = True
            st.success(f"✓ 知识库构建完成！共 {len(all_documents)} 个文本块")
            return True
        except Exception as e:
            st.error(f"✗ 构建知识库失败: {str(e)}")
            return False
    else:
        st.warning("未能处理任何文档")
        return False


def initialize_rag_chain():
    """初始化RAG链"""
    if st.session_state.rag_chain is None:
        try:
            st.session_state.rag_chain = RAGQAChain()
            st.session_state.rag_chain.initialize_llm()
            st.session_state.rag_chain.create_qa_chain()
        except Exception as e:
            st.error(f"✗ 初始化RAG链失败: {str(e)}")
            return False
    return True


def display_chat_history():
    """显示对话历史"""
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown("**问题:** " + question)
            st.markdown("**回答:** " + answer)
            st.markdown("---")


def main():
    """主函数"""
    init_session_state()

    # 标题
    st.markdown('<h1 class="main-header">📚 RAG智能问答系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于本地知识库的检索增强生成问答系统</p>', unsafe_allow_html=True)

    # 侧边栏 - 知识库管理
    with st.sidebar:
        st.header("📂 知识库管理")

        # 显示知识库状态
        stats = get_vector_store_stats()
        if stats.get("exists"):
            st.success("✓ 知识库已就绪")
            st.info(f"文档块数量: {stats.get('num_documents', 0)}")
        else:
            st.warning("⚠ 知识库为空")

        st.divider()

        # 文件上传
        st.subheader("上传文档")
        uploaded_files = st.file_uploader(
            "支持 PDF、DOCX 格式",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="上传与自然语言处理相关的文档以构建知识库"
        )

        if uploaded_files:
            st.write(f"已选择 {len(uploaded_files)} 个文件")

        # 构建知识库按钮
        if st.button("🔨 构建知识库", type="primary", disabled=not uploaded_files):
            with st.spinner("正在处理文档..."):
                process_uploaded_files(uploaded_files)
            st.rerun()

        st.divider()

        # 知识库操作
        st.subheader("知识库操作")
        if st.button("🗑 清空知识库"):
            import shutil
            if os.path.exists(config.VECTOR_STORE_PATH):
                shutil.rmtree(config.VECTOR_STORE_PATH)
            st.session_state.docs_uploaded = False
            st.session_state.rag_chain = None
            st.session_state.chat_history = []
            st.success("知识库已清空")
            st.rerun()

        st.divider()

        # 系统信息
        st.subheader("ℹ️ 系统信息")
        st.markdown(f"""
        - **大模型**: {config.LLM_MODEL}
        - **嵌入模型**: {config.EMBED_MODEL}
        - **Ollama地址**: {config.OLLAMA_BASE_URL}
        """)

    # 主内容区 - 问答界面
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 问答交互")

        # 检查知识库状态
        stats = get_vector_store_stats()
        if not stats.get("exists") or stats.get("num_documents", 0) == 0:
            st.warning("⚠️ 请先上传文档并构建知识库")
        else:
            # 初始化RAG链
            if not initialize_rag_chain():
                st.error("RAG链初始化失败，请检查Ollama服务")
            else:
                # 问答输入
                question = st.text_input(
                    "请输入您的问题：",
                    placeholder="例如：什么是自然语言处理？",
                    help="输入问题后点击'提问'按钮获取答案"
                )

                col_btn1, col_btn2 = st.columns([1, 4])
                with col_btn1:
                    ask_button = st.button("📤 提问", type="primary")

                # 清空对话按钮
                with col_btn2:
                    if st.button("🗑️ 清空对话"):
                        st.session_state.chat_history = []
                        st.rerun()

                # 处理提问
                if ask_button and question:
                    with st.spinner("思考中..."):
                        try:
                            result = st.session_state.rag_chain.answer(
                                question,
                                st.session_state.chat_history
                            )

                            # 添加到对话历史
                            st.session_state.chat_history.append(
                                (question, result["answer"])
                            )

                            # 显示答案
                            st.markdown("---")
                            st.markdown("**📝 回答:**")
                            st.markdown(result["answer"])

                            # 显示参考文档
                            if result.get("source_documents"):
                                with st.expander("📄 参考文档"):
                                    for i, doc in enumerate(result["source_documents"]):
                                        st.markdown(f"**文档 {i+1}:**")
                                        source = doc.metadata.get("source", "未知来源")
                                        st.markdown(f"来源: {source}")
                                        st.markdown(f"内容: {doc.page_content[:300]}...")
                                        st.markdown("---")

                        except Exception as e:
                            st.error(f"✗ 回答失败: {str(e)}")

    with col2:
        st.header("📜 对话历史")

        # 显示对话历史
        if st.session_state.chat_history:
            st.info(f"共 {len(st.session_state.chat_history)} 轮对话")

            # 显示最近的对话
            for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.container():
                    st.markdown(f"**Q:** {q[:50]}...")
                    st.markdown(f"**A:** {a[:100]}...")
                    st.markdown("---")
        else:
            st.info("暂无对话历史")

    # 底部信息
    st.divider()
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "RAG智能问答系统 | 基于 LangChain + Ollama + Streamlit"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
