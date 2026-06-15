# -*- coding: utf-8 -*-
"""
构建知识库脚本 - RAG智能问答系统
"""

from document_loader import DocumentLoader
from vector_store import VectorStoreManager

def build_knowledge_base():
    """构建知识库"""
    print("开始构建知识库...")
    
    # 初始化文档加载器
    loader = DocumentLoader()
    
    # 加载并分割文档
    print("加载文档...")
    documents = loader.process_folder()
    
    if not documents:
        print("没有加载到任何文档，请检查docs文件夹")
        return
    
    # 初始化向量数据库管理器
    manager = VectorStoreManager()
    
    # 创建向量数据库
    print("创建向量数据库...")
    manager.create_vectorstore(documents)
    
    # 获取统计信息
    stats = manager.get_collection_stats()
    print(f"\n知识库构建完成！")
    print(f"向量数据库状态: {stats}")

if __name__ == "__main__":
    build_knowledge_base()