# -*- coding: utf-8 -*-
"""
简化版RAG系统 - 使用内存向量存储
"""
import os
import sys

# 配置
DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "docs")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_txt_files(folder):
    """加载TXT文件"""
    documents = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append({"content": content, "source": file})
                    print(f"加载文档: {file}")
                except Exception as e:
                    print(f"加载失败 {file}: {e}")
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

def build_simple_knowledge_base():
    """构建简单的知识库"""
    print("=== 构建知识库 ===")
    
    # 加载文档
    print("\n1. 加载文档...")
    documents = load_txt_files(DOCS_FOLDER)
    if not documents:
        print("没有找到任何文档")
        return None
    
    # 分割文档
    print(f"\n2. 分割文档...")
    all_chunks = []
    for doc in documents:
        chunks = split_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append({"content": chunk, "source": doc["source"]})
    print(f"   分割成 {len(all_chunks)} 个文本块")
    
    # 保存到文件
    print("\n3. 保存知识库...")
    kb_file = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")
    with open(kb_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(all_chunks):
            f.write(f"=== Chunk {i+1} ({chunk['source']}) ===\n")
            f.write(chunk['content'])
            f.write("\n\n")
    print(f"   知识库已保存到: {kb_file}")
    
    return all_chunks

def search_knowledge_base(query, chunks, top_k=3):
    """简单的文本匹配搜索"""
    query_lower = query.lower()
    results = []
    
    for chunk in chunks:
        # 简单的关键词匹配
        score = sum(1 for word in query_lower.split() if word in chunk['content'].lower())
        if score > 0:
            results.append({"score": score, "content": chunk["content"], "source": chunk["source"]})
    
    # 按匹配度排序
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def answer_question(query, chunks):
    """回答问题"""
    print(f"\n=== 问答 ===")
    print(f"问题: {query}")
    
    # 搜索知识库
    results = search_knowledge_base(query, chunks)
    
    if not results:
        print("答案: 文档中未找到相关答案")
        return "文档中未找到相关答案"
    
    # 构建答案
    answer = "根据知识库，相关信息如下：\n\n"
    for i, result in enumerate(results):
        answer += f"来源: {result['source']}\n"
        answer += f"内容: {result['content'][:200]}...\n\n"
    
    print(f"答案:\n{answer}")
    return answer

if __name__ == "__main__":
    # 构建知识库
    chunks = build_simple_knowledge_base()
    
    if chunks:
        # 测试问答
        test_query = "什么是BERT模型？"
        answer_question(test_query, chunks)
        
        test_query2 = "什么是情感分析？"
        answer_question(test_query2, chunks)
        
        test_query3 = "什么是量子计算？"
        answer_question(test_query3, chunks)