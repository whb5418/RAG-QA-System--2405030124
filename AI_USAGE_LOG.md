# AI使用日志

本项目在开发过程中使用了AI编程辅助工具进行代码骨架生成、错误调试和提示词优化。

## AI使用记录

| 日期 | 使用内容 | 说明 | 备注 |
|------|----------|------|------|
| 2026-06-08 | 项目架构设计 | AI协助设计模块划分和代码结构 | 采用标准的RAG架构：文档加载→分块→向量化→检索→生成 |
| 2026-06-08 | 提示词优化 | AI协助优化RAG系统提示词 | 强调基于文档回答、拒绝回答无关问题 |
| 2026-06-08 | 错误调试 | AI协助排查Ollama连接问题 | 提供了连接检测和错误处理建议 |
| 2026-06-08 | 代码审查 | AI审查代码质量和安全性 | 检查硬编码、异常处理等 |

## AI辅助生成的关键代码片段

### 1. RAG提示词模板
```python
SYSTEM_PROMPT = """你是一个基于本地知识库的智能问答助手。你的任务是：
1. 基于提供的参考文档回答用户问题
2. 如果文档中没有相关信息，明确回答"文档中未找到相关答案"
3. 回答要准确、简洁、有条理
4. 如果需要，可以引用文档中的具体内容
请始终基于提供的参考文档来回答问题，不要编造信息。"""
```

### 2. 文档处理流程
```python
def process_folder(self, folder_path: str = None) -> List[Document]:
    documents = self.load_folder(folder_path)
    split_docs = self.split_documents(documents)
    return split_docs
```

### 3. 检索增强生成链
```python
combine_docs_chain = create_stuff_documents_chain(
    llm=self.llm,
    prompt=prompt,
    document_prompt=HumanMessagePromptTemplate.from_template("{page_content}")
)
self.qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain
)
```

## 向AI提问的示例

1. **Q**: 如何在LangChain中集成Ollama的嵌入模型？
   **A**: 使用 `OllamaEmbeddings` 类，指定model和base_url参数。

2. **Q**: 如何实现会话记忆功能？
   **A**: 使用 `st.session_state` 存储对话历史，在每次问答时传入chat_history参数。

3. **Q**: Chroma向量数据库如何持久化？
   **A**: 在 `Chroma.from_documents` 和 `Chroma` 初始化时指定 `persist_directory` 参数。

## 注意事项

- AI生成的代码需要理解后再使用
- 关键配置（如模型名称、路径）需要根据实际情况修改
- 异常处理部分需要根据实际情况调整
