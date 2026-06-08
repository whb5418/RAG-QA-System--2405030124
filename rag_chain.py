# -*- coding: utf-8 -*-
"""
RAG问答链模块 - RAG智能问答系统
整合检索器和Ollama大模型进行问答
"""

from typing import Optional, List, Dict
from langchain_ollama import ChatOllama
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
from vector_store import VectorStoreManager


class RAGQAChain:
    """RAG问答链类"""

    def __init__(self):
        """初始化RAG问答链"""
        self.vector_store_manager = VectorStoreManager()
        self.llm = None
        self.qa_chain = None

    def initialize_llm(self):
        """初始化Ollama大模型"""
        self.llm = ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.7,
            verbose=True
        )
        print(f"大模型初始化成功: {config.LLM_MODEL}")
        return self.llm

    def create_qa_chain(self):
        """
        创建问答链
        """
        if self.llm is None:
            self.initialize_llm()

        # 获取检索器
        retriever = self.vector_store_manager.get_retriever()

        # 构建提示词（包含context变量）
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", """请基于以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{input}

请根据参考文档回答：""")
        ])

        # 创建文档组合链
        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )

        # 创建检索链
        self.qa_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain
        )

        print("问答链创建成功")
        return self.qa_chain

    def answer(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None
    ) -> str:
        """
        回答问题

        Args:
            question: 用户问题
            chat_history: 对话历史

        Returns:
            答案字符串
        """
        if self.qa_chain is None:
            self.create_qa_chain()

        # 格式化对话历史
        formatted_history = []
        if chat_history:
            for q, a in chat_history:
                formatted_history.append({"role": "human", "content": q})
                formatted_history.append({"role": "assistant", "content": a})

        # 调用问答链
        result = self.qa_chain.invoke({
            "input": question,
            "chat_history": formatted_history
        })

        return result.get("answer", "")

    def get_relevant_documents(self, question: str) -> List[Document]:
        """
        获取与问题相关的文档

        Args:
            question: 用户问题

        Returns:
            相关的Document对象列表
        """
        return self.vector_store_manager.similarity_search(question)


if __name__ == "__main__":
    # 测试问答功能
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

    # 初始化大模型
    try:
        qa_chain.initialize_llm()
        qa_chain.create_qa_chain()
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        print("请确保Ollama服务已启动且模型已下载")
        return

    # 测试问题（与NLP相关）
    test_questions = [
        "什么是自然语言处理？",
        "RAG技术有哪些优点？",
        "Transformer模型的基本原理是什么？",
        "什么是词嵌入？",
        "请介绍一下注意力机制"
    ]

    print("\n" + "=" * 50)
    print("相关问题测试")
    print("=" * 50)

    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 40)
        try:
            answer = qa_chain.answer(question)
            # 限制输出长度
            if len(answer) > 500:
                answer = answer[:500] + "..."
            print(f"答案: {answer}")
        except Exception as e:
            print(f"回答失败: {str(e)}")