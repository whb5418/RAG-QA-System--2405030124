# -*- coding: utf-8 -*-
print("Test 1: Starting")

print("Test 2: Importing langchain_core.documents...")
from langchain_core.documents import Document
print("  OK")

print("Test 3: Importing langchain_chroma...")
from langchain_chroma import Chroma
print("  OK")

print("Test 4: Importing langchain_huggingface...")
from langchain_huggingface import HuggingFaceEmbeddings
print("  OK")

print("Test 5: Importing langchain_text_splitters...")
from langchain_text_splitters import RecursiveCharacterTextSplitter
print("  OK")

print("Test 6: Importing langchain_community.document_loaders...")
from langchain_community.document_loaders import PyPDFLoader
print("  OK")

print("All imports successful!")