# -*- coding: utf-8 -*-
"""
快速启动脚本 - RAG智能问答系统
一键启动Ollama服务（如果未启动）和Streamlit应用
"""

import subprocess
import sys
import time
import requests
import os

OLLAMA_BASE_URL = "http://localhost:11434"


def is_ollama_running():
    """检查Ollama服务是否运行"""
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=2)
        return response.status_code == 200
    except:
        return False


def start_ollama():
    """启动Ollama服务"""
    print("检查Ollama服务状态...")
    if is_ollama_running():
        print("✓ Ollama服务已在运行")
        return True

    print("正在启动Ollama服务...")
    try:
        # 在后台启动Ollama
        subprocess.Popen(
            ["ollama", "serve"],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        )
        # 等待服务启动
        for i in range(10):
            time.sleep(2)
            if is_ollama_running():
                print("✓ Ollama服务启动成功")
                return True
            print(f"  等待服务启动... ({i+1}/10)")
        print("✗ Ollama服务启动超时")
        return False
    except Exception as e:
        print(f"✗ 启动Ollama失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("RAG智能问答系统 - 启动脚本")
    print("=" * 50)

    # 启动Ollama
    if not start_ollama():
        print("\n请手动启动Ollama后重试，或运行: ollama serve")

    # 启动Streamlit
    print("\n启动Streamlit应用...")
    print("=" * 50)

    try:
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n应用已关闭")
    except Exception as e:
        print(f"启动失败: {str(e)}")


if __name__ == "__main__":
    main()
