# -*- coding: utf-8 -*-
import ollama
print("Ollama Python包导入成功")

# 尝试列出模型
try:
    models = ollama.list()
    print(f"可用模型: {models}")
except Exception as e:
    print(f"Error: {e}")