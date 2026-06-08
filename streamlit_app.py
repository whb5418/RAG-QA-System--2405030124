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

# Streamlit界面
st.title("🧠 RAG智能问答系统")

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