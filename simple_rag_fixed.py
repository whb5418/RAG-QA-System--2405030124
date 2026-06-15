# -*- coding: utf-8 -*-
"""
简化版RAG系统 - 修复编码问题
"""
import os
import sys

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "docs")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_txt_files(folder):
    """加载TXT文件，支持多种编码"""
    documents = []
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                content = None
                
                # 尝试多种编码
                for encoding in encodings:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            content = f.read()
                        break
                    except:
                        continue
                
                if content:
                    documents.append({"content": content, "source": file})
                    print(f"加载文档: {file}")
                else:
                    print(f"加载失败 {file}: 无法识别编码")
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
    """改进的文本匹配搜索"""
    results = []
    
    for chunk in chunks:
        # 使用in操作符进行简单匹配
        if query in chunk['content']:
            results.append({"score": 1, "content": chunk["content"], "source": chunk["source"]})
    
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
        answer += f"【来源: {result['source']}】\n"
        answer += f"{result['content'][:300]}...\n\n"
    
    print(f"答案:\n{answer}")
    return answer

if __name__ == "__main__":
    # 构建知识库
    chunks = build_knowledge_base()
    
    if chunks:
        # 测试问答
        print("\n" + "="*50)
        print("知识库构建成功！现在可以开始问答")
        print("="*50)
        
        while True:
            query = input("\n请输入您的问题（输入'退出'结束）：")
            if query == "退出":
                break
            answer_question(query, chunks)