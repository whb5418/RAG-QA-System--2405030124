# -*- coding: utf-8 -*-
"""
RAG智能问答系统 - Streamlit Web界面版本（优化版）
支持日常问题直接调用大模型，专业问题使用知识库
"""
import os
import sys

os.environ['OLLAMA_MODELS'] = 'ollama_models'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, '.')

import streamlit as st
import subprocess
import re
from vector_store import VectorStoreManager
from document_loader import DocumentLoader
import config

def clean_output(output):
    """清理Ollama输出中的控制字符"""
    output = output.decode('utf-8', errors='replace')
    output = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', output)
    output = re.sub(r'\[K', '', output)
    return output.strip()

def is_daily_question(query):
    """判断是否为日常问题（直接调用大模型）"""
    # 明确的日常问题关键词 - 这些问题应该直接调用大模型
    daily_keywords = ['天气', '气温', '温度', '时间', '几点', '日期', '今天', '明天', '昨天', '现在', '现在时间']
    return any(keyword in query for keyword in daily_keywords)

def direct_llm_answer(query):
    """直接调用大模型回答（不带知识库约束）"""
    prompt = f"""请回答以下问题：

{query}"""
    
    result = subprocess.run(
        ["ollama", "run", config.LLM_MODEL, prompt],
        capture_output=True,
        timeout=120
    )
    
    if result.returncode == 0:
        return clean_output(result.stdout)
    else:
        err = clean_output(result.stderr) if result.stderr else "未知错误"
        return f"错误: {err}"

def rag_answer(query):
    """使用知识库回答"""
    vs_manager = VectorStoreManager()
    docs = vs_manager.similarity_search(query, top_k=3)
    
    # 检查是否有相关文档
    if not docs or len(docs) == 0:
        return None, []
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""请基于以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{query}

请根据参考文档回答："""
    
    result = subprocess.run(
        ["ollama", "run", config.LLM_MODEL, prompt],
        capture_output=True,
        timeout=120
    )
    
    if result.returncode == 0:
        return clean_output(result.stdout), docs
    else:
        err = clean_output(result.stderr) if result.stderr else "未知错误"
        return f"错误: {err}", []

def get_answer(query):
    """获取回答（智能路由）"""
    # 判断是否为日常问题（如天气、时间等）
    if is_daily_question(query):
        # 日常问题直接调用大模型，不经过知识库
        return direct_llm_answer(query), "💡 日常问题（直接回答）"
    else:
        # 专业问题使用知识库
        answer, docs = rag_answer(query)
        if answer is None:
            return "抱歉，知识库中没有相关信息，无法回答这个问题。", "❌ 知识库无相关信息"
        return answer, "📚 基于知识库回答"

def build_knowledge_base():
    """构建知识库"""
    loader = DocumentLoader()
    documents = loader.process_folder()
    
    if not documents:
        return "没有加载到任何文档，请先上传文档"
    
    vs_manager = VectorStoreManager()
    vs_manager.create_vectorstore(documents)
    
    stats = vs_manager.get_collection_stats()
    return f"知识库构建成功！共存储 {stats.get('num_documents', 0)} 个文本块"

def add_document_to_knowledge_base(file_path):
    """添加单个文档到知识库"""
    loader = DocumentLoader()
    
    # 创建临时文件夹
    temp_folder = os.path.join(config.PROJECT_ROOT, "temp_docs")
    os.makedirs(temp_folder, exist_ok=True)
    
    try:
        # 加载文档
        documents = loader.load_document(file_path)
        
        if not documents:
            return "文档加载失败"
        
        # 添加到知识库
        vs_manager = VectorStoreManager()
        vs_manager.add_documents(documents)
        
        stats = vs_manager.get_collection_stats()
        return f"文档添加成功！知识库现有 {stats.get('num_documents', 0)} 个文本块"
    finally:
        # 清理临时文件
        if os.path.exists(temp_folder):
            import shutil
            shutil.rmtree(temp_folder)

# Streamlit界面
st.title("🧠 RAG智能问答系统")

# 侧边栏 - 知识库管理
with st.sidebar:
    st.header("📁 知识库管理")
    
    # 显示知识库状态
    vs_manager = VectorStoreManager()
    stats = vs_manager.get_collection_stats()
    
    if stats.get("exists"):
        st.success(f"✅ 知识库已构建")
        st.info(f"文档数量: {stats.get('num_documents', 0)}")
    else:
        st.warning("⚠️ 知识库未构建")
    
    # 上传文档区域
    uploaded_files = st.file_uploader(
        "上传文档到知识库",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # 保存上传的文件到临时位置
            temp_path = os.path.join(config.PROJECT_ROOT, "docs", uploaded_file.name)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"已保存: {uploaded_file.name}")
    
    # 构建/更新知识库按钮
    if st.button("🔄 构建/更新知识库"):
        with st.spinner("正在构建知识库..."):
            result = build_knowledge_base()
            st.info(result)
    
    # 清空知识库按钮
    if st.button("🗑️ 清空知识库"):
        vs_manager.clear_vectorstore()
        st.success("知识库已清空")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message:
            st.caption(message["source"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            answer, source = get_answer(prompt)
            st.markdown(answer)
            st.caption(source)
    
    # 添加AI回答
    st.session_state.messages.append({"role": "assistant", "content": answer, "source": source})