# -*- coding: utf-8 -*-
"""
向量数据库管理模块 - RAG智能问答系统
使用Chroma向量数据库存储和检索文档嵌入
"""

import os
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config


class VectorStoreManager:
    """向量数据库管理器"""

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None
    ):
        """
        初始化向量数据库管理器

        Args:
            persist_directory: 向量数据库持久化路径
            collection_name: 集合名称
        """
        self.persist_directory = persist_directory or config.VECTOR_STORE_PATH
        self.collection_name = collection_name or config.CHROMA_DB_NAME
        self._embeddings = None
        self._vectorstore = None
    
    @property
    def embeddings(self):
        """延迟初始化嵌入模型"""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            print("HuggingFace嵌入模型初始化成功")
        return self._embeddings

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        创建新的向量数据库

        Args:
            documents: Document对象列表

        Returns:
            Chroma向量数据库实例
        """
        # 如果已存在，先删除
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"已删除旧的向量数据库: {self.persist_directory}")

        # 创建新的向量数据库
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

        print(f"向量数据库创建成功，共存储 {len(documents)} 个文本块")
        return self._vectorstore

    def get_vectorstore(self) -> Optional[Chroma]:
        """
        获取已存在的向量数据库

        Returns:
            Chroma向量数据库实例，如果不存在则返回None
        """
        if self._vectorstore is not None:
            return self._vectorstore

        if not os.path.exists(self.persist_directory):
            return None

        try:
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print(f"向量数据库加载成功")
            return self._vectorstore
        except Exception as e:
            print(f"加载向量数据库失败: {str(e)}")
            return None

    def get_retriever(self, top_k: int = None):
        """
        获取检索器

        Args:
            top_k: 返回的最相关文档数量

        Returns:
            检索器对象
        """
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            raise ValueError("向量数据库不存在，请先创建")

        k = top_k or config.TOP_K
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, top_k: int = None) -> List[Document]:
        """
        相似性搜索

        Args:
            query: 查询字符串
            top_k: 返回的最相关文档数量

        Returns:
            最相关的Document对象列表
        """
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            raise ValueError("向量数据库不存在，请先创建")

        k = top_k or config.TOP_K
        return vectorstore.similarity_search(query, k=k)

    def get_collection_stats(self) -> dict:
        """
        获取向量数据库统计信息

        Returns:
            包含统计信息的字典
        """
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            return {
                "exists": False,
                "num_documents": 0
            }

        try:
            collection = vectorstore._collection
            return {
                "exists": True,
                "num_documents": collection.count(),
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            print(f"获取统计信息失败: {str(e)}")
            return {
                "exists": True,
                "num_documents": 0,
                "error": str(e)
            }

    def add_documents(self, documents: List[Document]) -> None:
        """
        向现有向量数据库添加文档

        Args:
            documents: Document对象列表
        """
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            # 如果不存在，创建新的
            self.create_vectorstore(documents)
        else:
            vectorstore.add_documents(documents)
            print(f"已添加 {len(documents)} 个文档到向量数据库")

    def clear_vectorstore(self):
        """
        清除向量数据库
        """
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"已清除向量数据库: {self.persist_directory}")
        self._vectorstore = None
        self._embeddings = None


if __name__ == "__main__":
    # 测试检索功能
    print("测试检索功能...")
    manager = VectorStoreManager()

    # 检查向量数据库是否存在
    stats = manager.get_collection_stats()
    print(f"向量数据库状态: {stats}")

    if not stats.get("exists"):
        print("向量数据库不存在，请先构建知识库")
        exit(0)

    # 测试检索
    query = "什么是自然语言处理？"
    print(f"\n测试查询: {query}")

    try:
        results = manager.similarity_search(query)
        print(f"检索到 {len(results)} 个相关文档")
        for i, doc in enumerate(results):
            print(f"\n--- 结果 {i+1} ---")
            print(doc.page_content[:200] + "...")
    except Exception as e:
        print(f"检索失败: {str(e)}")