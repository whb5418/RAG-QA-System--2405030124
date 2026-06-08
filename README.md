# RAG-QA-System-姓名-学号

# RAG智能问答系统

基于本地知识库的检索增强生成（Retrieval-Augmented Generation）智能问答系统，使用Ollama部署的本地大模型、LangChain框架和Streamlit构建。

## 项目简介

本项目实现了一个基于本地知识库的RAG智能问答系统，能够：
- 加载和解析PDF、DOCX文档
- 将文档分块并向量化存储到Chroma向量数据库
- 基于检索增强生成技术回答用户问题
- 提供友好的Web交互界面

## 环境要求

- Python 3.10+
- Windows 10/11
- Ollama（用于运行本地大模型）

## 安装步骤

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 2. 安装Ollama

访问 [Ollama官网](https://ollama.com/download) 下载并安装Windows版本。

### 3. 下载模型

打开命令行，执行以下命令下载所需的模型：

```bash
# 下载大语言模型（deepseek-r1:7b 或 qwen2:7b）
ollama pull deepseek-r1:7b

# 下载嵌入模型
ollama pull nomic-embed-text
```

### 4. 启动Ollama服务

```bash
ollama serve
```

## 使用说明

### 启动Web应用

```bash
streamlit run app.py
```

浏览器将自动打开 http://localhost:8501

### 使用流程

1. **上传文档**: 在左侧上传PDF或DOCX格式的文档（建议上传与自然语言处理相关的文档）
2. **构建知识库**: 点击"构建知识库"按钮，系统将自动解析文档、分块并向量化
3. **开始问答**: 在输入框中输入问题，点击"提问"获取答案
4. **查看对话历史**: 右侧面板显示对话历史

## 关键技术点

### RAG流程

```
用户问题 → 检索（向量相似度匹配）→ 获取相关文档 → 结合文档与问题 → LLM生成答案
```

### 所用模型

- **大语言模型**: deepseek-r1:7b（由Ollama提供）
- **嵌入模型**: nomic-embed-text（用于将文本向量化）

### 嵌入方式

使用Ollama内置的嵌入模型将文本分块转换为向量，存储到Chroma向量数据库中。检索时，通过余弦相似度找到最相关的文档块。

## 项目结构

```
.
├── app.py                 # Streamlit Web应用主文件
├── config.py              # 配置文件
├── document_loader.py     # 文档加载与处理模块
├── vector_store.py        # 向量数据库管理模块
├── rag_chain.py           # RAG问答链模块
├── test_ollama.py         # Ollama API测试脚本
├── build_exe.py           # PyInstaller打包脚本
├── requirements.txt       # Python依赖列表
├── docs/                  # 文档存储目录（上传的文档放这里）
├── vector_store/          # 向量数据库存储目录
└── README.md              # 项目说明文档
```

## 项目效果截图

### 1. Web应用主界面
![主界面](screenshots/main_interface.png)

### 2. 文档上传与知识库构建
![知识库构建](screenshots/knowledge_base.png)

### 3. 问答功能演示
![问答演示](screenshots/qa_demo.png)

## 已知问题与改进方向

### 已知问题

- 首次加载大模型可能需要较长时间
- 文档解析对扫描版PDF支持有限

### 改进方向

- [ ] 支持更多文档格式（PPT、Excel等）
- [ ] 添加夜间模式
- [ ] 支持批量上传
- [ ] 添加导出问答记录功能
- [ ] 优化检索算法，提升相关性

## AI使用日志

本项目在开发过程中使用了AI编程辅助工具进行代码骨架生成、错误调试和提示词优化。

| 日期 | 使用内容 | 说明 |
|------|----------|------|
| 2026-06-08 | 项目架构设计 | AI协助设计模块划分和代码结构 |
| 2026-06-08 | 提示词优化 | AI协助优化RAG系统提示词 |
| 2026-06-08 | 错误调试 | AI协助排查Ollama连接问题 |

## License

MIT License
