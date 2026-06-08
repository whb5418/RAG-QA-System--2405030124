# -*- coding: utf-8 -*-
"""
构建脚本 - 将Streamlit应用打包为exe
使用PyInstaller将应用打包成独立的Windows可执行文件
"""

import os
import sys
import subprocess
import shutil

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 构建输出目录
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")

# 资源目录
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")


def clean_build_dirs():
    """清理构建目录"""
    print("清理构建目录...")
    for dir_path in [DIST_DIR, BUILD_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"  已删除: {dir_path}")


def install_pyinstaller():
    """安装PyInstaller"""
    print("检查PyInstaller...")
    try:
        import PyInstaller
        print("  PyInstaller已安装")
        return True
    except ImportError:
        print("  正在安装PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("  PyInstaller安装完成")
        return True


def create_spec_file():
    """创建PyInstaller spec文件"""
    spec_content = '''# -*- coding: utf-8 -*-
import sys
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(SPEC))

block_cipher = None

# 收集所有需要的资源
datas = [
    (os.path.join(PROJECT_ROOT, 'config.py'), '.'),
    (os.path.join(PROJECT_ROOT, 'document_loader.py'), '.'),
    (os.path.join(PROJECT_ROOT, 'vector_store.py'), '.'),
    (os.path.join(PROJECT_ROOT, 'rag_chain.py'), '.'),
]

a = Analysis(
    [os.path.join(PROJECT_ROOT, 'app.py')],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'streamlit',
        'langchain',
        'langchain_community',
        'langchain_ollama',
        'chromadb',
        'pypdf',
        'python_docx',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RAG-QA-System',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RAG-QA-System',
)
'''

    spec_path = os.path.join(PROJECT_ROOT, "app.spec")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_content)
    print(f"  已创建spec文件: {spec_path}")
    return spec_path


def build_exe():
    """构建exe文件"""
    print("\n开始构建exe文件...")
    print("这可能需要几分钟时间，请耐心等待...")

    spec_file = os.path.join(PROJECT_ROOT, "app.spec")

    try:
        subprocess.check_call([
            "pyinstaller",
            "--clean",
            spec_file
        ])
        print("\n✓ 构建成功！")
        print(f"  exe文件位置: {DIST_DIR}\\RAG-QA-System\\RAG-QA-System.exe")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 构建失败: {str(e)}")
        return False


def create_readme_for_exe():
    """创建exe使用说明"""
    readme_content = """# RAG智能问答系统 - exe版本使用说明

## 依赖环境

在运行exe之前，请确保已安装以下软件：

1. **Ollama**: 下载地址 https://ollama.com/download
2. **模型下载**: 打开命令行执行以下命令：
   ```
   ollama pull deepseek-r1:7b
   ollama pull nomic-embed-text
   ```
3. **启动Ollama服务**:
   ```
   ollama serve
   ```

## 运行方法

1. 双击 `RAG-QA-System.exe` 启动应用
2. 浏览器会自动打开 http://localhost:8501
3. 如果浏览器没有自动打开，请手动访问上述地址

## 使用步骤

1. **上传文档**: 在左侧上传PDF或DOCX格式的文档
2. **构建知识库**: 点击"构建知识库"按钮
3. **开始问答**: 在输入框中输入问题，点击"提问"

## 注意事项

- 首次运行可能需要几分钟时间来启动
- 请确保Ollama服务正在运行
- 如果遇到问题，请查看命令行窗口的错误信息
"""

    readme_path = os.path.join(DIST_DIR, "RAG-QA-System", "README.txt")
    os.makedirs(os.path.dirname(readme_path), exist_ok=True)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  已创建使用说明: {readme_path}")


def main():
    """主函数"""
    print("=" * 50)
    print("RAG智能问答系统 - 构建脚本")
    print("=" * 50)

    # 清理旧构建
    clean_build_dirs()

    # 安装PyInstaller
    if not install_pyinstaller():
        print("PyInstaller安装失败")
        return False

    # 创建spec文件
    create_spec_file()

    # 构建exe
    if build_exe():
        # 创建使用说明
        create_readme_for_exe()
        print("\n" + "=" * 50)
        print("构建完成！")
        print("=" * 50)
    else:
        print("\n构建失败，请检查错误信息")
        return False

    return True


if __name__ == "__main__":
    main()
