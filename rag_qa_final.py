# -*- coding: utf-8 -*-
"""
RAG智能问答系统 - 最终版本 (使用subprocess)
"""
import subprocess
import os
import sys
import re

# 设置环境变量
os.environ['OLLAMA_MODELS'] = 'ollama_models'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sys.path.insert(0, '.')

import config
from vector_store import VectorStoreManager

def clean_output(output):
    """清理Ollama输出中的控制字符"""
    output = output.decode('utf-8', errors='replace')
    output = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', output)
    output = re.sub(r'\[K', '', output)
    return output.strip()

print("=" * 60)
print("    RAG智能问答系统 v1.0")
print("=" * 60)

# 检查向量数据库
vs_manager = VectorStoreManager()
stats = vs_manager.get_collection_stats()
print(f"\n知识库状态: {'已就绪' if stats.get('exists') and stats.get('num_documents', 0) > 0 else '未就绪'}")
print(f"文档数量: {stats.get('num_documents', 0)}")

if not stats.get('exists') or stats.get('num_documents', 0) == 0:
    print("\n错误: 知识库未构建")
    input("\n按回车键退出...")
    sys.exit(1)

print("\n" + "=" * 60)
print("系统已就绪！可以开始提问")
print("=" * 60)

while True:
    question = input("\n请输入您的问题（输入 'exit' 退出）：")
    if question.lower() == 'exit':
        break
    if not question.strip():
        continue
    
    print("\n检索相关文档...")
    
    # 获取相关文档
    docs = vs_manager.similarity_search(question, top_k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    if not context:
        print("未找到相关文档")
        continue
    
    # 构建prompt
    prompt = f"""请基于以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{question}

请根据参考文档回答："""
    
    print("生成回答...")
    
    # 使用subprocess调用Ollama
    try:
        result = subprocess.run(
            ["ollama", "run", config.LLM_MODEL, prompt],
            capture_output=True,
            timeout=120
        )
        
        if result.returncode == 0:
            answer = clean_output(result.stdout)
            print(f"\n答案:\n{answer}")
        else:
            error_msg = clean_output(result.stderr) if result.stderr else "未知错误"
            print(f"\n错误: {error_msg}")
            
    except subprocess.TimeoutExpired:
        print("\n错误: 回答超时")
    except Exception as e:
        print(f"\n错误: {e}")

print("\n感谢使用RAG智能问答系统！")