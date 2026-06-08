# -*- coding: utf-8 -*-
"""
RAG问答链模块 - RAG智能问答系统
整合检索器和Ollama大模型进行问答
"""

from typing import Optional, List, Dict
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

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

    def create_qa_chain(self) -> ConversationalRetrievalChain:
        """
        创建问答链

        Returns:
            ConversationalRetrievalChain实例
        """
        if self.llm is None:
            self.initialize_llm()

        # 获取检索器
        retriever = self.vector_store_manager.get_retriever()

        # 构建提示词
        system_prompt = SystemMessagePromptTemplate.from_template(config.SYSTEM_PROMPT)
        human_prompt = HumanMessagePromptTemplate.from_template(
            "请根据以下参考文档回答问题。\n\n"
            "参考文档：\n"
            "{context}\n\n"
            "用户问题：{input}"
        )
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

        # 创建文档组合链
        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_prompt=HumanMessagePromptTemplate.from_template("{page_content}")
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
    ) -> Dict:
        """
        回答问题

        Args:
            question: 用户问题
            chat_history: 对话历史，格式为[(问题, 回答), ...]

        Returns:
            包含答案和上下文的字典
        """
        if self.qa_chain is None:
            self.create_qa_chain()

        # 格式化对话历史
        formatted_history = []
        if chat_history:
            for q, a in chat_history:
                formatted_history.append(f"用户：{q}\n助手：{a}")

        # 调用问答链
        result = self.qa_chain.invoke({
            "input": question,
            "chat_history": formatted_history if formatted_history else []
        })

        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", []),
            "source_documents": result.get("source_documents", [])
        }

    def get_relevant_documents(self, question: str) -> List[Document]:
        """
        获取与问题相关的文档

        Args:
            question: 用户问题

        Returns:
            相关的Document对象列表
        """
        return self.vector_store_manager.similarity_search(question)


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

    # 无关问题测试
    unrelated_questions = [
        "今天天气怎么样？",
        "如何做红烧肉？"
    ]

    print("\n" + "=" * 50)
    print("相关问题测试")
    print("=" * 50)

    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 40)
        try:
            result = qa_chain.answer(question)
            answer = result["answer"]
            # 限制输出长度
            if len(answer) > 500:
                answer = answer[:500] + "..."
            print(f"答案: {answer}")
            print(f"参考文档数: {len(result['source_documents'])}")
        except Exception as e:
            print(f"回答失败: {str(e)}")

    print("\n" + "=" * 50)
    print("无关问题测试")
    print("=" * 50)

    for question in unrelated_questions:
        print(f"\n问题: {question}")
        print("-" * 40)
        try:
            result = qa_chain.answer(question)
            answer = result["answer"]
            if len(answer) > 500:
                answer = answer[:500] + "..."
            print(f"答案: {answer}")
        except Exception as e:
            print(f"回答失败: {str(e)}")


if __name__ == "__main__":
    test_qa()
