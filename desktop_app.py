# -*- coding: utf-8 -*-
"""
RAG智能问答系统 - PyQt5桌面应用版
"""
import sys
import os
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QFileDialog, QListWidget,
    QProgressBar, QLabel, QMessageBox, QSplitter, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon

# 导入业务逻辑模块
import config
from document_loader import DocumentLoader
from vector_store import VectorStoreManager
from rag_chain import RAGQAChain

# 全局变量
qa_chain_instance = None
vector_store_instance = None

class WorkerThread(QThread):
    """工作线程，用于执行耗时操作"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int)
    
    def __init__(self, task_type, documents=None, query=None, parent=None):
        super().__init__(parent)
        self.task_type = task_type
        self.documents = documents
        self.query = query
    
    def run(self):
        try:
            if self.task_type == 'build_knowledge':
                self.build_knowledge_base()
            elif self.task_type == 'query':
                self.execute_query()
        except Exception as e:
            self.finished.emit(False, str(e))
    
    def build_knowledge_base(self):
        try:
            self.progress.emit(10)
            
            loader = DocumentLoader()
            documents = loader.load_folder()
            
            if not documents:
                self.finished.emit(False, "未找到任何文档")
                return
            
            global vector_store_instance, qa_chain_instance
            
            vector_store_instance = VectorStoreManager()
            self.progress.emit(20)
            
            split_docs = loader.split_documents(documents)
            self.progress.emit(40)
            
            vector_store_instance.create_vectorstore(split_docs)
            self.progress.emit(80)
            
            # 不在这里初始化RAG问答链（会加载大模型，占用大量内存）
            # 只在第一次提问时初始化
            qa_chain_instance = None
            self.progress.emit(100)
            
            self.finished.emit(True, "知识库构建完成")
        except Exception as e:
            self.finished.emit(False, str(e))
    
    def execute_query(self):
        try:
            global qa_chain_instance, vector_store_instance
            
            # 如果vector_store_instance未创建，尝试加载已存在的知识库
            if not vector_store_instance:
                if os.path.exists(config.VECTOR_STORE_PATH):
                    vector_store_instance = VectorStoreManager()
                else:
                    self.finished.emit(False, "知识库未就绪，请先构建知识库")
                    return
            
            # 如果RAG链未初始化，先初始化
            if not qa_chain_instance:
                qa_chain_instance = RAGQAChain()
            
            answer = qa_chain_instance.answer(self.query)
            self.finished.emit(True, answer)
        except Exception as e:
            self.finished.emit(False, str(e))

class ChatMessageWidget(QFrame):
    """聊天消息组件"""
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
            }
        """)
        
        layout = QHBoxLayout()
        label = QLabel("User:" if is_user else "AI:")
        label.setFont(QFont("Arial", 14))
        
        text_edit = QTextEdit()
        text_edit.setPlainText(text)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("""
            QTextEdit {
                background: transparent;
                border: none;
                font-size: 14px;
            }
        """)
        
        if is_user:
            self.setStyleSheet("QFrame { background-color: #E8F4FD; }")
            layout.addStretch()
            layout.addWidget(text_edit)
            layout.addWidget(label)
        else:
            self.setStyleSheet("QFrame { background-color: #F0F2F5; }")
            layout.addWidget(label)
            layout.addWidget(text_edit)
            layout.addStretch()
        
        self.setLayout(layout)

class MainWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.check_existing_knowledge_base()
    
    def init_ui(self):
        """初始化UI"""
        # 设置窗口标题和大小
        self.setWindowTitle("RAG智能问答系统")
        self.setGeometry(100, 100, 1000, 600)
        self.setMinimumSize(800, 500)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧面板 - 文档管理
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 文档列表
        self.doc_list = QListWidget()
        
        # 上传按钮
        upload_btn = QPushButton("上传文档")
        upload_btn.clicked.connect(self.upload_documents)
        upload_btn.setStyleSheet(self.get_button_style())
        
        # 构建知识库按钮
        build_btn = QPushButton("构建知识库")
        build_btn.clicked.connect(self.build_knowledge_base)
        build_btn.setStyleSheet(self.get_button_style())
        
        # 清除知识库按钮
        clear_btn = QPushButton("清除知识库")
        clear_btn.clicked.connect(self.clear_knowledge_base)
        clear_btn.setStyleSheet(self.get_button_style())
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # 状态标签
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setStyleSheet("color: #666;")
        
        left_layout.addWidget(QLabel("<b>文档列表</b>"))
        left_layout.addWidget(self.doc_list)
        left_layout.addWidget(upload_btn)
        left_layout.addWidget(build_btn)
        left_layout.addWidget(clear_btn)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch()
        
        splitter.addWidget(left_panel)
        splitter.setSizes([250, 750])
        
        # 右侧面板 - 聊天区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 聊天区域
        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_area.setWidget(self.chat_widget)
        
        # 输入区域
        input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("请输入您的问题...")
        self.input_edit.returnPressed.connect(self.send_query)
        
        send_btn = QPushButton("提问")
        send_btn.clicked.connect(self.send_query)
        send_btn.setStyleSheet(self.get_button_style())
        
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(send_btn)
        
        right_layout.addWidget(QLabel("<b>智能问答</b>"))
        right_layout.addWidget(self.chat_area)
        right_layout.addLayout(input_layout)
        
        splitter.addWidget(right_panel)
        
        # 初始化文档列表
        self.refresh_doc_list()
    
    def get_button_style(self):
        return """
            QPushButton {
                background-color: #4A90D9;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3A7BC8;
            }
            QPushButton:pressed {
                background-color: #2A6BB8;
            }
        """
    
    def check_existing_knowledge_base(self):
        """检查是否已存在向量数据库（不加载embedding模型）"""
        global vector_store_instance
        
        # 只检查目录是否存在，不创建VectorStoreManager实例
        if os.path.exists(config.VECTOR_STORE_PATH):
            # 检查是否有数据文件
            has_files = any(os.listdir(config.VECTOR_STORE_PATH))
            if has_files:
                self.status_label.setText("状态: 知识库就绪")
                self.add_message("AI:", "检测到已存在的知识库，我已准备好回答问题！")
                # 延迟创建vector_store_instance，在第一次提问时创建
                vector_store_instance = None
    
    def refresh_doc_list(self):
        """刷新文档列表"""
        self.doc_list.clear()
        if os.path.exists(config.DOCS_FOLDER):
            for file in os.listdir(config.DOCS_FOLDER):
                if file.endswith(('.pdf', '.docx', '.doc')):
                    self.doc_list.addItem(file)
    
    def upload_documents(self):
        """上传文档"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择文档", "", "文档文件 (*.pdf *.docx *.doc)"
        )
        
        if files:
            os.makedirs(config.DOCS_FOLDER, exist_ok=True)
            for file in files:
                file_name = os.path.basename(file)
                dest_path = os.path.join(config.DOCS_FOLDER, file_name)
                
                # 复制文件
                with open(file, 'rb') as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())
            
            QMessageBox.information(self, "成功", "已上传 " + str(len(files)) + " 个文档")
            self.refresh_doc_list()
    
    def build_knowledge_base(self):
        """构建知识库"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("状态: 正在构建知识库...")
        
        # 禁用按钮并保存引用
        self.build_btn_ref = self.sender()
        self.build_btn_ref.setEnabled(False)
        
        # 创建工作线程
        self.worker = WorkerThread('build_knowledge')
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_build_finished)
        self.worker.start()
    
    def update_progress(self, value):
        """更新进度"""
        self.progress_bar.setValue(value)
    
    def on_build_finished(self, success, message):
        """构建完成处理"""
        self.progress_bar.setVisible(False)
        # 重新启用按钮
        if hasattr(self, 'build_btn_ref') and self.build_btn_ref:
            self.build_btn_ref.setEnabled(True)
        
        if success:
            self.status_label.setText("状态: 知识库就绪")
            QMessageBox.information(self, "成功", message)
            self.add_message("AI:", "知识库构建完成！我已经准备好了，可以开始提问。")
        else:
            self.status_label.setText("状态: 构建失败")
            QMessageBox.warning(self, "失败", message)
    
    def clear_knowledge_base(self):
        """清除知识库"""
        global qa_chain_instance, vector_store_instance
        
        reply = QMessageBox.question(
            self, "确认", "确定要清除知识库吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                if vector_store_instance:
                    vector_store_instance.clear_vectorstore()
                
                qa_chain_instance = None
                vector_store_instance = None
                
                # 清空聊天记录
                while self.chat_layout.count() > 0:
                    item = self.chat_layout.takeAt(0)
                    item.widget().deleteLater()
                
                self.status_label.setText("状态: 就绪")
                QMessageBox.information(self, "成功", "知识库已清除")
            except Exception as e:
                QMessageBox.warning(self, "失败", str(e))
    
    def send_query(self):
        """发送查询"""
        query = self.input_edit.text().strip()
        if not query:
            return
        
        # 添加用户消息
        self.add_message("User:", query)
        self.input_edit.clear()
        
        # 检查知识库是否就绪
        if not qa_chain_instance:
            QMessageBox.warning(self, "警告", "请先构建知识库")
            return
        
        self.status_label.setText("状态: 正在思考...")
        
        # 创建工作线程
        self.query_worker = WorkerThread('query', query=query)
        self.query_worker.finished.connect(self.on_query_finished)
        self.query_worker.start()
    
    def on_query_finished(self, success, message):
        """查询完成处理"""
        self.status_label.setText("状态: 就绪")
        
        if success:
            self.add_message("AI:", message)
        else:
            QMessageBox.warning(self, "失败", message)
    
    def add_message(self, icon, text):
        """添加消息到聊天区域"""
        is_user = icon == "User:"
        msg_widget = ChatMessageWidget(text, is_user)
        self.chat_layout.addWidget(msg_widget)
        
        # 滚动到底部
        QApplication.processEvents()
        scroll_bar = self.chat_area.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

def main():
    app = QApplication(sys.argv)
    
    # 设置全局样式
    app.setStyleSheet("""
        QWidget {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            font-size: 14px;
        }
        QMainWindow {
            background-color: #F5F7FA;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
