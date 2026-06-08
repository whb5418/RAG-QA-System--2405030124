# -*- coding: utf-8 -*-
"""
文档加载与处理模块 - RAG智能问答系统
支持PDF和DOCX文档的加载与文本提取
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


class DocumentLoader:
    """文档加载器类"""

    def __init__(self, docs_folder: str = None):
        """
        初始化文档加载器

        Args:
            docs_folder: 文档文件夹路径，默认为config中的DOCS_FOLDER
        """
        self.docs_folder = docs_folder or config.DOCS_FOLDER
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path: str) -> List[Document]:
        """
        加载单个文档

        Args:
            file_path: 文档文件路径

        Returns:
            Document对象列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_extension}")

        documents = loader.load()
        return documents

    def load_folder(self, folder_path: str = None) -> List[Document]:
        """
        加载文件夹中的所有文档

        Args:
            folder_path: 文件夹路径，默认为DOCS_FOLDER

        Returns:
            所有加载的Document对象列表
        """
        folder = folder_path or self.docs_folder

        if not os.path.exists(folder):
            raise FileNotFoundError(f"文件夹不存在: {folder}")

        all_documents = []
        supported_extensions = [".pdf", ".docx", ".doc"]

        for root, dirs, files in os.walk(folder):
            for file in files:
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension in supported_extensions:
                    file_path = os.path.join(root, file)
                    try:
                        documents = self.load_document(file_path)
                        all_documents.extend(documents)
                        print(f"成功加载文档: {file}")
                    except Exception as e:
                        print(f"加载文档失败 {file}: {str(e)}")

        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档分割成小块

        Args:
            documents: Document对象列表

        Returns:
            分割后的Document对象列表
        """
        if not documents:
            return []

        split_docs = self.text_splitter.split_documents(documents)
        print(f"文档分割完成，共 {len(split_docs)} 个文本块")
        return split_docs

    def process_folder(self, folder_path: str = None) -> List[Document]:
        """
        加载并分割文件夹中的所有文档

        Args:
            folder_path: 文件夹路径

        Returns:
            分割后的Document对象列表
        """
        documents = self.load_folder(folder_path)
        split_docs = self.split_documents(documents)
        return split_docs


def get_document_stats(documents: List[Document]) -> dict:
    """
    获取文档统计信息

    Args:
        documents: Document对象列表

    Returns:
        包含统计信息的字典
    """
    if not documents:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_characters": 0
        }

    total_documents = len(set(doc.metadata.get("source", "") for doc in documents))
    total_chunks = len(documents)
    total_characters = sum(len(doc.page_content) for doc in documents)

    return {
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "total_characters": total_characters
    }


if __name__ == "__main__":
    # 测试文档加载
    print("测试文档加载功能...")
    loader = DocumentLoader()

    try:
        docs = loader.load_folder()
        print(f"共加载 {len(docs)} 个文档")

        split_docs = loader.split_documents(docs)
        print(f"分割后共 {len(split_docs)} 个文本块")

        stats = get_document_stats(split_docs)
        print(f"统计信息: {stats}")
    except Exception as e:
        print(f"测试失败: {str(e)}")