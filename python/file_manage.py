import requests
import os
import faiss
import numpy as np
import time
import shutil
from typing import List, Tuple, Dict, Any
from urllib.parse import quote

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Loaders
from langchain_community.document_loaders import (
    CSVLoader, 
    TextLoader, 
    UnstructuredExcelLoader, 
    PyPDFLoader
)
from app import BASE_DIR
####
# ---------------- CONFIGURATION ---------------- #
API_BASE_URL = "http://3.6.203.180:7501"
GET_ALL_DOCS_URL = f"{API_BASE_URL}/documents/getAllDocuments"
DOWNLOAD_DOC_URL = f"{API_BASE_URL}/documents/"

DOCUMENTS_DIR = "documents"
EMBEDDINGS_DIR = "Embeddings"

# Ensure directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Initialize Embedding Model (adjust model_name as needed)
# Using a small, fast model for demonstration
_embedder_model = HuggingFaceEmbeddings(model_name=os.path.join(BASE_DIR , "models" , "all-MiniLM-L6-v2"),
            model_kwargs={ "device": "cpu",
            "local_files_only": True   # ← THIS FIXES SERVER ISSUE
        },
        encode_kwargs={"normalize_embeddings": True},)

# ---------------- HELPER: FILE HANDLING ---------------- #

def get_file_key(file_path: str) -> str:
    """Generates a safe filename key based on the file path."""
    return os.path.splitext(os.path.basename(file_path))[0]

def load_single_document(file_path: str) -> List[Document]:
    """
    Loads data and SPLITS it into chunks so the embedding model sees everything.
    """
    ext = os.path.splitext(file_path)[1].lower()
    raw_docs = []
    
    try:
        if ext == ".csv":
            loader = CSVLoader(file_path)
            raw_docs = loader.load()
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
            raw_docs = loader.load()
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
        elif ext == ".xlsx":
            loader = UnstructuredExcelLoader(file_path)
            raw_docs = loader.load()
        else:
            return []
            
        # --- FIX: SPLIT TEXT INTO CHUNKS ---
        # chunk_size=500 ensures the text fits into the model's context window
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=90,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        docs = text_splitter.split_documents(raw_docs)
        
        # Add source metadata
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
            
        print(f"📄 Loaded {os.path.basename(file_path)}: Split into {len(docs)} chunks.")
        return docs

    except Exception as e:
        print(f"❌ Error loading file {file_path}: {e}")
        return []
    
# ---------------- API INTERACTION ---------------- #

def fetch_api_file_list() -> List[str]:
    """Fetch list of available document NAMES from external API."""
    try:
        print(f"🌐 Fetching list from: {GET_ALL_DOCS_URL}")
        response = requests.get(GET_ALL_DOCS_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('success') == 'success' and 'data' in data:
            # Assuming 'data' is a list of filenames like ['file1.pdf', 'file2.csv']
            return data['data'] 
        return []
    except Exception as e:
        print(f"❌ Error fetching API list: {e}")
        return []

def download_file(filename: str) -> bool:
    """Downloads a file from API to the local documents folder."""
    try:
        encoded_name = quote(filename)
        url = f"{DOWNLOAD_DOC_URL}{encoded_name}"
        save_path = os.path.join(DOCUMENTS_DIR, filename)
        
        print(f"⬇️ Downloading: {filename}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Saved: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

# ---------------- SYNC LOGIC (DELETE & DOWNLOAD) ---------------- #

def sync_local_storage_with_api() -> List[str]:
    
    api_files = fetch_api_file_list()
    if not api_files:
        print("⚠️ API returned no files or failed. Aborting sync to prevent data loss.")

    
    api_files = set(f['name'] for f in api_files) 
    api_files_set = api_files
    
    # --- PHASE 1: DELETE ORPHANS ---
    local_files = os.listdir(DOCUMENTS_DIR)
    for f in local_files:
        if f not in api_files_set:
            full_path = os.path.join(DOCUMENTS_DIR, f)
            if os.path.isfile(full_path):
                print(f"🗑️ Deleting orphan document: {f} (Not in API)")
                os.remove(full_path)

    # --- PHASE 2: DOWNLOAD MISSING ---
    valid_paths = []
    for f_name in api_files:
        full_path = os.path.join(DOCUMENTS_DIR, f_name)
        
        # If file doesn't exist locally, download it
        if not os.path.exists(full_path):
            success = download_file(f_name)
            if success:
                valid_paths.append(full_path)
        else:
            valid_paths.append(full_path)
            
    return valid_paths

# ---------------- EMBEDDING CACHE LOGIC ---------------- #

def cleanup_orphaned_embeddings(valid_file_paths: List[str]):
    """Deletes .npy/.index files for documents that are no longer valid."""
    # Create a set of "Expected" keys (e.g., "myfile")
    valid_keys = {get_file_key(f) for f in valid_file_paths}
    
    for fname in os.listdir(EMBEDDINGS_DIR):
        if fname.endswith("_faiss.index") or fname.endswith("_docs.npy"):
            # logic: 'myfile_faiss.index' -> 'myfile'
            base_key = fname.replace("_faiss.index", "").replace("_docs.npy", "")
            
            if base_key not in valid_keys:
                print(f"🗑️ Removing orphaned embedding cache: {fname}")
                os.remove(os.path.join(EMBEDDINGS_DIR, fname))

def load_or_create_embeddings(file_path: str, embedder) -> Tuple[Any, List[Document]]:
    base_name = get_file_key(file_path)
    index_path = os.path.join(EMBEDDINGS_DIR, f"{base_name}_faiss.index")
    docs_path = os.path.join(EMBEDDINGS_DIR, f"{base_name}_docs.npy")

    # A. Try Load
    if os.path.exists(index_path) and os.path.exists(docs_path):
        try:
            index = faiss.read_index(index_path)
            docs = np.load(docs_path, allow_pickle=True).tolist()
            if index.ntotal == len(docs):
                return index, docs
        except Exception:
            pass # Fallback to rebuild

    # B. Create (Build)
    print(f"🔨 Building embeddings for: {os.path.basename(file_path)}")
    
    # This now returns CHUNKED documents
    chunked_docs = load_single_document(file_path)
    
    if not chunked_docs:
        return None, []

    # Create Embeddings
    text_embeddings = embedder.embed_documents([d.page_content for d in chunked_docs])
    text_embeddings_np = np.array(text_embeddings).astype("float32")
    
    # --- FIX: USE INNER PRODUCT (COSINE SIMILARITY) ---
    # Since embeddings are normalized, Inner Product == Cosine Similarity
    dimension = text_embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension) 
    index.add(text_embeddings_np)
    
    # C. Save
    faiss.write_index(index, index_path)
    np.save(docs_path, np.array(chunked_docs, dtype=object))
    
    return index, chunked_docs
# ---------------- MAIN PIPELINE ---------------- #

def get_synced_vectorstore() -> FAISS:
    """
    Orchestrates the entire flow:
    Sync Files -> Cleanup Embeddings -> Load/Create Embeddings -> Merge to VectorStore
    """
    print("🔄 STARTING SYNC PROCESS...")
    
    # 1. Sync Documents (Download/Delete)
    valid_files = sync_local_storage_with_api() 
    valid_files.append('documents\\real_estate.txt') 
    print(valid_files)
    print(f"✅ Document Sync Complete. {len(valid_files)} valid files locally.")

    # 2. Cleanup Embeddings (Delete orphans)
    # cleanup_orphaned_embeddings(valid_files)

    # 3. Load/Create Embeddings for valid files
    all_vectors = []
    all_docs = []
    
    for file_path in valid_files:
        index, docs = load_or_create_embeddings(file_path, _embedder_model)
        
        if index is not None and len(docs) > 0:
            # Reconstruct vectors from FAISS index to merge later
            ntotal = index.ntotal
            vecs = np.zeros((ntotal, index.d), dtype='float32')
            for i in range(ntotal):
                vecs[i] = index.reconstruct(i)
            
            all_vectors.append(vecs)
            all_docs.extend(docs)

    if not all_docs:
        print("⚠️ No documents available to index.")
        # Return an empty store or handle gracefully
        empty_embed = _embedder_model.embed_query("test")
        dim = len(empty_embed)
        return FAISS(
            embedding_function=_embedder_model,
            index=faiss.IndexFlatL2(dim),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    # 4. Merge into Main VectorStore
    print(f"🚀 Merging {len(all_docs)} chunks into main Memory...")
    combined_vectors = np.vstack(all_vectors)
    dimension = combined_vectors.shape[1]
    
    # --- FIX: USE INNER PRODUCT HERE TOO ---
    main_index = faiss.IndexFlatIP(dimension)
    main_index.add(combined_vectors) 
    
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}
    for i, doc in enumerate(all_docs):
        docstore.add({str(i): doc})
        index_to_docstore_id[i] = str(i)

    vectorstore = FAISS(
        embedding_function=_embedder_model,
        index=main_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    return vectorstore
