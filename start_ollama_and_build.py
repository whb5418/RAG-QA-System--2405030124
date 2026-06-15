# -*- coding: utf-8 -*-
"""
启动Ollama服务并构建知识库
"""
import subprocess
import time
import sys

def start_ollama_service():
    """启动Ollama服务"""
    print("正在启动Ollama服务...")
    try:
        # 使用ollama serve启动服务
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        print("Ollama服务已启动")
        return process
    except Exception as e:
        print(f"启动Ollama服务失败: {e}")
        return None

def check_ollama_service():
    """检查Ollama服务是否可用"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def build_knowledge_base():
    """构建知识库"""
    print("正在构建知识库...")
    
    from document_loader import DocumentLoader
    from vector_store import VectorStoreManager
    
    loader = DocumentLoader()
    documents = loader.process_folder()
    
    if not documents:
        print("没有加载到任何文档")
        return False
    
    vs_manager = VectorStoreManager()
    vs_manager.create_vectorstore(documents)
    
    stats = vs_manager.get_collection_stats()
    print(f"知识库构建成功！文档数: {stats.get('num_documents', 0)}")
    return True

if __name__ == "__main__":
    # 检查Ollama服务
    if not check_ollama_service():
        print("Ollama服务未运行，尝试启动...")
        # 尝试启动Ollama服务
        ollama_process = start_ollama_service()
        
        # 等待服务启动
        print("等待Ollama服务启动...")
        for _ in range(30):
            if check_ollama_service():
                print("Ollama服务已就绪")
                break
            time.sleep(1)
        else:
            print("Ollama服务启动超时")
            sys.exit(1)
    
    # 构建知识库
    build_knowledge_base()
    print("\n知识库构建完成！")