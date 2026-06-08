# -*- coding: utf-8 -*-
"""
配置文件 - RAG智能问答系统
"""
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 文档存储目录
DOCS_FOLDER = os.path.join(PROJECT_ROOT, "docs")

# 向量数据库存储路径
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "vector_store")

# Chroma数据库名称
CHROMA_DB_NAME = "rag_knowledge_base"

# 文本分块参数
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ollama配置
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "phi3:mini"
EMBED_MODEL = "nomic-embed-text"

# 检索参数
TOP_K = 3

# 系统提示词
SYSTEM_PROMPT = """你是一个基于本地知识库的智能问答助手。你的任务是：
1. 基于提供的参考文档回答用户问题
2. 如果文档中没有相关信息，明确回答"文档中未找到相关答案"
3. 回答要准确、简洁、有条理
4. 如果需要，可以引用文档中的具体内容

请始终基于提供的参考文档来回答问题，不要编造信息。"""

# 检索提示词模板
RAG_PROMPT_TEMPLATE = """请基于以下参考文档回答用户问题。如果文档中没有相关信息，请明确说明。

参考文档：
{context}

用户问题：{question}

请根据参考文档回答："""