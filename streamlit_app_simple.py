# -*- coding: utf-8 -*-
"""
RAG智能问答系统 - Streamlit Web界面（简化版）
使用简单的文本匹配实现知识库问答，支持无关问题调用大模型
"""
import os
import streamlit as st
import subprocess
import re
import config

# 配置
DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "docs")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
KB_FILE = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")

# 全局变量存储知识库
if 'knowledge_chunks' not in st.session_state:
    st.session_state.knowledge_chunks = []

def load_txt_files(folder):
    """加载TXT文件，支持多种编码"""
    documents = []
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                content = None
                
                for encoding in encodings:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            content = f.read()
                        break
                    except:
                        continue
                
                if content:
                    documents.append({"content": content, "source": file})
                else:
                    st.warning(f"无法读取文件: {file}")
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=200):
    """简单的文本分块"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if start > 0:
            chunk = text[start - chunk_overlap:end]
        chunks.append(chunk)
        start = end
    return chunks

def build_knowledge_base():
    """构建知识库"""
    st.info("正在加载文档...")
    
    # 加载文档
    documents = load_txt_files(DOCS_FOLDER)
    if not documents:
        return "没有找到任何文档"
    
    # 分割文档
    all_chunks = []
    for doc in documents:
        chunks = split_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append({"content": chunk, "source": doc["source"]})
    
    # 保存到全局状态
    st.session_state.knowledge_chunks = all_chunks
    
    # 保存到文件
    with open(KB_FILE, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(all_chunks):
            f.write(f"=== Chunk {i+1} ({chunk['source']}) ===\n")
            f.write(chunk['content'])
            f.write("\n\n")
    
    return f"知识库构建成功！共存储 {len(all_chunks)} 个文本块"

def search_knowledge_base(query, chunks, top_k=3):
    """搜索知识库"""
    results = []
    
    for chunk in chunks:
        if query in chunk['content']:
            results.append({"score": 1, "content": chunk["content"], "source": chunk["source"]})
    
    return results[:top_k]

def clean_output(output):
    """清理Ollama输出中的控制字符"""
    if isinstance(output, bytes):
        output = output.decode('utf-8', errors='replace')
    output = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', output)
    output = re.sub(r'\[K', '', output)
    return output.strip()

def call_llm(query):
    """调用大模型回答问题"""
    try:
        result = subprocess.run(
            ["ollama", "run", config.LLM_MODEL, query],
            capture_output=True,
            timeout=120
        )
        
        if result.returncode == 0:
            return clean_output(result.stdout)
        else:
            return f"大模型调用失败: {clean_output(result.stderr) if result.stderr else '未知错误'}"
    except subprocess.TimeoutExpired:
        return "回答超时，请稍后重试"
    except FileNotFoundError:
        return "Ollama未安装或未启动，请先安装Ollama"

def get_answer(query):
    """获取回答（优先知识库，其次大模型）"""
    chunks = st.session_state.knowledge_chunks
    
    if not chunks:
        # 知识库未构建，直接调用大模型
        answer = call_llm(query)
        return answer, "🤖 大模型直接回答（知识库未构建）"
    
    # 搜索知识库
    results = search_knowledge_base(query, chunks)
    
    if not results:
        # 知识库无匹配，调用大模型
        answer = call_llm(query)
        return answer, "🤖 大模型直接回答（知识库无匹配）"
    
    # 构建答案
    answer = "根据知识库，相关信息如下：\n\n"
    for i, result in enumerate(results):
        answer += f"**来源**: {result['source']}\n\n"
        answer += f"{result['content'][:500]}...\n\n"
    
    return answer, "📚 基于知识库回答"

# Streamlit界面
st.title("🧠 RAG智能问答系统")

# 侧边栏 - 知识库管理
with st.sidebar:
    st.header("📁 知识库管理")
    
    # 显示知识库状态
    if st.session_state.knowledge_chunks:
        st.success(f"✅ 知识库已构建")
        st.info(f"文本块数量: {len(st.session_state.knowledge_chunks)}")
    else:
        st.warning("⚠️ 知识库未构建")
    
    # 上传文档区域
    uploaded_files = st.file_uploader(
        "上传文档到知识库",
        type=["txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # 保存上传的文件到docs文件夹
            temp_path = os.path.join(DOCS_FOLDER, uploaded_file.name)
            os.makedirs(DOCS_FOLDER, exist_ok=True)
            
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
        st.session_state.knowledge_chunks = []
        if os.path.exists(KB_FILE):
            os.remove(KB_FILE)
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
        with st.spinner("正在搜索知识库..."):
            answer, source = get_answer(prompt)
            st.markdown(answer)
            st.caption(source)
    
    # 添加AI回答
    st.session_state.messages.append({"role": "assistant", "content": answer, "source": source})