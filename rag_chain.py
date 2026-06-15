# -*- coding: utf-8 -*-
"""
RAG问答链模块 - RAG智能问答系统
使用ConversationalRetrievalChain整合检索器和Ollama大模型进行问答
"""

from typing import Optional, List, Dict
from langchain_ollama import ChatOllama
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

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
        创建问答链 - 使用ConversationalRetrievalChain
        """
        if self.llm is None:
            self.initialize_llm()

        # 获取检索器
        retriever = self.vector_store_manager.get_retriever()

        # 构建系统提示词 - 核心要求：基于文档回答，无信息时返回"文档中未找到相关答案"
        system_prompt = SystemMessagePromptTemplate.from_template(
            config.SYSTEM_PROMPT
        )

        # 构建人类消息提示词
        human_prompt = HumanMessagePromptTemplate.from_template(
            """请基于以下参考文档回答用户问题。

参考文档：
{context}

用户问题：{question}

请根据参考文档回答："""
        )

        # 组合提示词
        prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            human_prompt
        ])

        # 创建ConversationalRetrievalChain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )

        print("ConversationalRetrievalChain问答链创建成功")
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
            chat_history: 对话历史 [(问题1, 答案1), (问题2, 答案2), ...]

        Returns:
            答案字符串
        """
        if self.qa_chain is None:
            self.create_qa_chain()

        # 格式化对话历史为LangChain消息格式
        formatted_history = []
        if chat_history:
            for q, a in chat_history:
                formatted_history.append(HumanMessage(content=q))
                formatted_history.append(AIMessage(content=a))

        # 调用问答链
        result = self.qa_chain.invoke({
            "question": question,
            "chat_history": formatted_history
        })

        answer = result.get("answer", "")
        
        # 确保如果没有找到答案，返回指定字符串
        if not answer or "未找到" not in answer and len(answer.strip()) < 10:
            # 检查源文档是否有相关内容
            source_docs = result.get("source_documents", [])
            if not source_docs:
                answer = "文档中未找到相关答案"
        
        return answer

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
    print("RAG问答系统测试 - ConversationalRetrievalChain")
    print("=" * 50)

    qa_chain = RAGQAChain()

    # 检查向量数据库
    stats = qa_chain.vector_store_manager.get_collection_stats()
    print(f"\n向量数据库状态: {stats}")

    if not stats.get("exists") or stats.get("num_documents", 0) == 0:
        print("错误: 向量数据库为空或不存在，请先构建知识库")
        exit(0)

    # 初始化大模型
    try:
        qa_chain.initialize_llm()
        qa_chain.create_qa_chain()
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        print("请确保Ollama服务已启动且模型已下载")
        exit(0)

    # 测试问题
    test_questions = [
        "什么是自然语言处理？",
        "RAG技术有哪些优点？",
        "Transformer模型的基本原理是什么？",
        "什么是词嵌入？",
        "请介绍一下注意力机制",
        "今天天气怎么样？"  # 这个问题文档中应该没有相关信息
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
