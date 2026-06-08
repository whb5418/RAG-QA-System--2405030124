# -*- coding: utf-8 -*-
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(SPEC))

block_cipher = None

datas = [
    (os.path.join(PROJECT_ROOT, 'config.py'), '.'),
    (os.path.join(PROJECT_ROOT, 'document_loader.py'), '.'),
    (os.path.join(PROJECT_ROOT, 'vector_store.py'), '.'),
    (os.path.join(PROJECT_ROOT, 'rag_chain.py'), '.'),
    (os.path.join(PROJECT_ROOT, 'docs'), 'docs'),
    (os.path.join(PROJECT_ROOT, 'chroma_db'), 'chroma_db'),
]

a = Analysis(
    [os.path.join(PROJECT_ROOT, 'desktop_app.py')],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'langchain',
        'langchain_community',
        'langchain_ollama',
        'langchain_text_splitters',
        'langchain_core',
        'chromadb',
        'pypdf',
        'python_docx',
        'requests',
        'tiktoken',
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
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
    console=False,
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
