# -*- coding: utf-8 -*-
"""
Ollama API测试脚本 - RAG智能问答系统
验证Ollama服务和大模型是否正常工作
"""

import requests
import json

# Ollama服务地址
OLLAMA_BASE_URL = "http://localhost:11434"

# 模型名称
LLM_MODEL = "deepseek-r1:7b"
EMBED_MODEL = "nomic-embed-text"


def test_ollama_service():
    """测试Ollama服务是否运行"""
    print("测试Ollama服务...")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama服务正常运行")
            models = response.json().get("models", [])
            print(f"  已下载的模型: {len(models)} 个")
            for model in models:
                print(f"    - {model.get('name', 'unknown')}")
            return True
        else:
            print(f"✗ Ollama服务返回异常状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到Ollama服务")
        print("  请确保Ollama已安装并正在运行")
        print("  在Windows上，可以通过以下命令启动Ollama:")
        print("    ollama serve")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


def test_llm_inference():
    """测试大模型推理"""
    print("\n测试大模型推理...")
    print(f"  使用模型: {LLM_MODEL}")

    try:
        payload = {
            "model": LLM_MODEL,
            "prompt": "请用一句话介绍自己",
            "stream": False
        }
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "")
            print("✓ 大模型推理正常")
            print(f"  模型回复: {answer[:200]}...")
            return True
        else:
            print(f"✗ 大模型推理失败，状态码: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print("✗ 大模型推理超时")
        print("  如果使用的是7B以上模型，可能需要较长时间")
        print("  建议使用更小的模型或等待模型加载完成")
        return False
    except Exception as e:
        print(f"✗ 大模型推理失败: {str(e)}")
        return False


def test_embeddings():
    """测试嵌入模型"""
    print("\n测试嵌入模型...")
    print(f"  使用模型: {EMBED_MODEL}")

    try:
        payload = {
            "model": EMBED_MODEL,
            "prompt": "这是测试文本"
        }
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            embedding = result.get("embedding", [])
            print("✓ 嵌入模型正常")
            print(f"  嵌入向量维度: {len(embedding)}")
            return True
        else:
            print(f"✗ 嵌入模型失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 嵌入模型失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("Ollama API 测试")
    print("=" * 50)

    results = {
        "service": test_ollama_service(),
        "embeddings": False,
        "llm": False
    }

    if results["service"]:
        results["embeddings"] = test_embeddings()
        results["llm"] = test_llm_inference()

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    print(f"Ollama服务: {'✓ 通过' if results['service'] else '✗ 失败'}")
    print(f"嵌入模型: {'✓ 通过' if results['embeddings'] else '✗ 失败'}")
    print(f"大模型推理: {'✓ 通过' if results['llm'] else '✗ 失败'}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ 所有测试通过！系统已准备好运行RAG问答系统")
    else:
        print("\n✗ 部分测试失败，请检查上述错误信息")
        print("\n安装和启动指南:")
        print("1. 安装Ollama: https://ollama.com/download")
        print("2. 下载模型:")
        print(f"   ollama pull {LLM_MODEL}")
        print(f"   ollama pull {EMBED_MODEL}")
        print("3. 启动Ollama服务:")
        print("   ollama serve")


if __name__ == "__main__":
    main()
