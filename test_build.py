# -*- coding: utf-8 -*-
import sys
print('Python版本:', sys.version)
print('开始构建知识库...')

from document_loader import DocumentLoader
print('DocumentLoader导入成功')

loader = DocumentLoader()
print('DocumentLoader初始化成功')

documents = loader.process_folder()
print(f'加载了 {len(documents)} 个文档')

if not documents:
    print('没有加载到任何文档')
else:
    print('开始创建向量数据库...')
    from vector_store import VectorStoreManager
    print('VectorStoreManager导入成功')
    
    vs_manager = VectorStoreManager()
    print('VectorStoreManager初始化成功')
    
    vs_manager.create_vectorstore(documents)
    print('向量数据库创建成功')
    
    stats = vs_manager.get_collection_stats()
    print(f'构建完成! 文档数: {stats.get("num_documents", 0)}')
    print(f'统计信息: {stats}')