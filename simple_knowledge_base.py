# -*- coding: utf-8 -*-
"""
简单的知识库构建脚本
"""
import os
import sys
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 配置
DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "docs")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "vector_store")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_txt_files(folder):
    """加载文件夹中的所有TXT文件"""
    documents = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append(Document(page_content=content, metadata={"source": file}))
                    print(f"加载文档: {file}")
                except Exception as e:
                    print(f"加载失败 {file}: {e}")
    return documents

def simple_split(documents, chunk_size=1000, chunk_overlap=200):
    """简单的文本分块"""
    split_docs = []
    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if start > 0:
                chunk = text[start - chunk_overlap:end]
            split_docs.append(Document(page_content=chunk, metadata=metadata))
            start = end
    return split_docs

def build_knowledge_base():
    """构建知识库"""
    print("开始构建知识库...")
    
    # 加载文档
    print("加载文档...")
    documents = load_txt_files(DOCS_FOLDER)
    if not documents:
        print("没有找到任何文档")
        return False
    print(f"共加载 {len(documents)} 个文档")
    
    # 分割文档
    print("分割文档...")
    split_docs = simple_split(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"分割成 {len(split_docs)} 个文本块")
    
    # 创建向量数据库
    print("创建向量数据库...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("嵌入模型初始化成功")
    
    # 删除旧的向量数据库
    import shutil
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
        print("已删除旧的向量数据库")
    
    # 创建新的向量数据库
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    print("向量数据库创建成功")
    
    # 保存
    vectorstore.persist()
    
    # 统计
    count = vectorstore._collection.count()
    print(f"知识库构建完成！共存储 {count} 个文本块")
    return True

if __name__ == "__main__":
    build_knowledge_base()