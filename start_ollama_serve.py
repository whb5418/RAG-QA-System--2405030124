# -*- coding: utf-8 -*-
import subprocess
import time
import os

# 尝试查找ollama可执行文件
def find_ollama():
    possible_paths = [
        "ollama",
        os.path.expanduser("~/.ollama/ollama.exe"),
        "C:\\Program Files\\Ollama\\ollama.exe",
        "C:\\ollama\\ollama.exe"
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return path
        except:
            continue
    return None

ollama_path = find_ollama()
if ollama_path:
    print(f"找到Ollama: {ollama_path}")
    
    # 启动Ollama服务
    print("启动Ollama服务...")
    process = subprocess.Popen(
        [ollama_path, "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    # 等待服务启动
    print("等待服务启动...")
    time.sleep(5)
    
    # 检查服务状态
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("Ollama服务启动成功！")
        else:
            print(f"服务状态: {response.status_code}")
    except Exception as e:
        print(f"无法连接到Ollama服务: {e}")
else:
    print("未找到Ollama可执行文件")