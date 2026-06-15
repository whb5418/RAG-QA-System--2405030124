# -*- coding: utf-8 -*-
print('Step 1: Starting...')

print('Step 2: Importing os...')
import os

print('Step 3: Importing config...')
import config

print('Step 4: Importing langchain_core.documents...')
from langchain_core.documents import Document

print('Step 5: Importing RecursiveCharacterTextSplitter...')
from langchain_text_splitters import RecursiveCharacterTextSplitter

print('Step 6: Importing PyPDFLoader...')
from langchain_community.document_loaders import PyPDFLoader

print('Step 7: All imports successful!')

print('Step 8: Creating DocumentLoader...')
class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_folder(self):
        folder = config.DOCS_FOLDER
        all_docs = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in [".txt"]:
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    all_docs.append(Document(page_content=content, metadata={"source": file_path}))
                    print(f"Loaded: {file}")
        return all_docs
    
    def split_documents(self, docs):
        return self.text_splitter.split_documents(docs)

print('Step 9: Creating instance...')
loader = DocumentLoader()
print('Step 10: Loading documents...')
docs = loader.load_folder()
print(f'Step 11: Loaded {len(docs)} docs')

print('Step 12: Splitting...')
split_docs = loader.split_documents(docs)
print(f'Step 13: Split into {len(split_docs)} chunks')

print('Done!')