# -*- coding: utf-8 -*-
print('Step 1: Starting...')
import sys
print('Step 2: sys imported')
print('Python版本:', sys.version)

print('Step 3: Trying to import DocumentLoader...')
try:
    from document_loader import DocumentLoader
    print('Step 4: DocumentLoader imported successfully')
except Exception as e:
    print(f'Error importing DocumentLoader: {e}')
    sys.exit(1)

print('Step 5: Creating DocumentLoader instance...')
try:
    loader = DocumentLoader()
    print('Step 6: DocumentLoader instance created')
except Exception as e:
    print(f'Error creating DocumentLoader: {e}')
    sys.exit(1)

print('Step 7: Loading documents...')
try:
    documents = loader.process_folder()
    print(f'Step 8: Loaded {len(documents)} documents')
except Exception as e:
    print(f'Error loading documents: {e}')
    sys.exit(1)

if not documents:
    print('No documents loaded!')
    sys.exit(0)

print('Step 9: Trying to import VectorStoreManager...')
try:
    from vector_store import VectorStoreManager
    print('Step 10: VectorStoreManager imported')
except Exception as e:
    print(f'Error importing VectorStoreManager: {e}')
    sys.exit(1)

print('Step 11: Creating VectorStoreManager...')
try:
    vs_manager = VectorStoreManager()
    print('Step 12: VectorStoreManager created')
except Exception as e:
    print(f'Error creating VectorStoreManager: {e}')
    sys.exit(1)

print('Step 13: Creating vectorstore...')
try:
    vs_manager.create_vectorstore(documents)
    print('Step 14: Vectorstore created')
except Exception as e:
    print(f'Error creating vectorstore: {e}')
    sys.exit(1)

print('Step 15: Getting stats...')
try:
    stats = vs_manager.get_collection_stats()
    print(f'Step 16: Stats: {stats}')
except Exception as e:
    print(f'Error getting stats: {e}')
    sys.exit(1)

print('Done!')