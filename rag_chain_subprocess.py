# -*- coding: utf-8 -*-
"""
RAG问答链模块 - 使用subprocess调用Ollama
"""

import subprocess
import os
from typing import Optional, List

import config
from vector_store import VectorStoreManager

# 设置环境变量
os.environ['OLLAMA_MODELS'] = 'c:\\Users\\lenovopc\\Documents\\trae_projects\\2\\ollama_models'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class RAGQAChain:
    """RAG问答链类"""

    def __init__(self):
        """初始化RAG问答链"""
        self.vector_store_manager = VectorStoreManager()

    def get_relevant_context(self, question: str) -> str:
        """获取与问题相关的文档内容"""
        docs = self.vector_store_manager.similarity_search(question, top_k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

    def answer(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None
    ) -> str:
        """
        回答问题
        """
        # 获取相关文档
        context = self.get_relevant_context(question)
        
        if not context:
            return "文档中未找到相关答案"

        # 构建prompt
        prompt = f"""请基于以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{question}

请根据参考文档回答："""

        # 使用subprocess调用ollama
        try:
            result = subprocess.run(
                ["C:\\Users\\lenovopc\\AppData\\Local\\Programs\\Ollama\\ollama.exe", "run", config.LLM_MODEL, prompt],
                capture_output=True,
                timeout=120
            )
            
            if result.returncode == 0:
                # 解码输出，处理控制字符
                output = result.stdout.decode('utf-8', errors='replace')
                # 移除ANSI转义序列和控制字符
                import re
                output = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', output)
                output = re.sub(r'\[K', '', output)
                output = output.strip()
                return output
            else:
                error_msg = result.stderr.decode('utf-8', errors='replace') if result.stderr else "未知错误"
                return f"回答失败: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return "回答超时"
        except Exception as e:
            return f"回答失败: {str(e)}"


def test_qa():
    """测试问答功能"""
    print("=" * 50)
    print("RAG问答系统测试")
    print("=" * 50)

    qa_chain = RAGQAChain()

    # 检查向量数据库
    stats = qa_chain.vector_store_manager.get_collection_stats()
    print(f"\n向量数据库状态: {stats}")

    if not stats.get("exists") or stats.get("num_documents", 0) == 0:
        print("错误: 向量数据库为空或不存在，请先构建知识库")
        return

    # 测试问题
    test_questions = [
        "什么是自然语言处理？",
        "RAG技术有哪些优点？",
        "什么是词嵌入？",
    ]

    print("\n" + "=" * 50)
    print("测试问答")
    print("=" * 50)

    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 40)
        try:
            answer = qa_chain.answer(question)
            print(f"答案: {answer}")
            # 写入文件验证
            with open('qa_test_result.txt', 'a', encoding='utf-8') as f:
                f.write(f"问题: {question}\n答案: {answer}\n\n")
        except Exception as e:
            print(f"回答失败: {str(e)}")


if __name__ == "__main__":
    test_qa()