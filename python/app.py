import os
import json
import re
import warnings
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter, defaultdict
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Request , Body
from fastapi.responses import StreamingResponse , JSONResponse
import sys, time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import requests
from symspellpy import SymSpell
# from langchain.retrievers.multi_query import MultiQueryRetriever
import httpx
from difflib import SequenceMatcher
from llama_cpp import Llama
import asyncio 
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# ========================================
# Paths and globals
# ========================================
DATA_DIR = "documents"
EMB_PATH = "Embeddings"
META_FILE = os.path.join(EMB_PATH, "meta.json")
DATA_FILE = "documents/data1.txt"
OUTPUT_DICT = "dictionary.txt"
CONTEXT_FILE = "documents/context.txt"
MAX_CONTEXT_TURNS = 4
REFERENCE_CLEANUP="documents/cleanup_reference.txt"

SESSION_STATE = defaultdict(dict)  # per-session state: {"active_society": str}



EMB_PATH = "Embeddings"
EMB_FILE = os.path.join(EMB_PATH, "embeddings.npy")   # numeric embeddings
TEXT_FILE = os.path.join(EMB_PATH, "texts.json")      # original document text
META_FILE = os.path.join(EMB_PATH, "meta.json")       # timestamp + info

# ========================================
# Utilities
# ========================================
def get_file_timestamp(path: str) -> float:
    return os.path.getmtime(path)

def embeddings_exist() -> bool:
    return (
        os.path.exists(os.path.join(EMB_PATH, "index.faiss"))
        and os.path.exists(os.path.join(EMB_PATH, "index.pkl"))
        and os.path.exists(META_FILE)
    )

def save_vectorstore(vectorstore: FAISS, timestamp: float) -> None:
    os.makedirs(EMB_PATH, exist_ok=True)
    vectorstore.save_local(EMB_PATH)
    json.dump({"timestamp": timestamp}, open(META_FILE, "w"))

def load_existing_vectorstore(embeddings):
    print("🔄 Loading existing FAISS vector store...")
    return FAISS.load_local(EMB_PATH, embeddings, allow_dangerous_deserialization=True)

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def get_session_id(request: Request) -> str:
    return request.headers.get("x-session-id") or request.client.host or "anon"

# ========================================
# Society name detection and tagging
# ========================================
SOCIETY_KEYWORDS = (
    "society","apartment","apartments","residency","residences","heights",
    "homes","estate","OneWorld developers","Galaxy Developers","Sobha Limited","Panchshil Realty","Lodha Group","galaxy","apex"
)
NAME_RX = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+(%s)\b" % "|".join(SOCIETY_KEYWORDS),
    re.IGNORECASE,
)

def extract_society_name(text: str) -> Optional[str]:
    m = NAME_RX.search(text)
    if not m:
        return None
    return (" ".join([m.group(1), m.group(2)])).strip().title()

def load_documents(file_path: str = DATA_FILE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    last_name = None
    for d in chunks:
        found = extract_society_name(d.page_content)
        if found:
            last_name = found
        d.metadata = d.metadata or {}
        d.metadata["society"] = last_name or "unknown"
    return chunks

# ========================================
# Embeddings, Vectorstore, Retriever
# ========================================
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="models/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    


def build_vectorstore() -> FAISS:
    print("🔨 Rebuilding embeddings & index...")
    docs = load_documents(DATA_FILE)
    embeddings = create_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    timestamp = get_file_timestamp(DATA_FILE)
    save_vectorstore(vectorstore, timestamp)
    return vectorstore

def get_vectorstore_and_retriever():
    embeddings = create_embeddings()
    current_timestamp = get_file_timestamp(DATA_FILE)

    if embeddings_exist():
        stored_meta = json.load(open(META_FILE))
        if current_timestamp == stored_meta.get("timestamp", 0):
            vs = load_existing_vectorstore(embeddings)
            print("✅ Using cached vectorstore")
            return vs, vs.as_retriever(search_kwargs={"k": 12})
        print("⚠️ Data changed → refreshing embedding store")

    vs = build_vectorstore()
    return vs, vs.as_retriever(search_kwargs={"k": 12})

# def create_multiquery_retriever(vectorstore, llm):
    
#     return MultiQueryRetriever.from_llm(
#         retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
#         llm=llm  # you already have an LLM function
#     )


vectorstore, base_retriever = get_vectorstore_and_retriever()


def collect_known_societies(vs: FAISS) -> set[str]:
    names = set()
    # access underlying dict
    for _id, doc in vs.docstore._dict.items():
        s = doc.metadata.get("society") if doc.metadata else None
        if s and s != "unknown":
            names.add(str(s))
    return names

KNOWN_SOCIETIES = collect_known_societies(vectorstore)

# ========================================
# Reranker
# ========================================
def create_reranker(embeddings, top_k: int = 3, threshold: float = 0.3) -> RunnableLambda:
    def _rerank(inputs):
        docs = inputs["docs"]
        query = inputs["question"]
        if not docs:
            return []

        q_vec = np.array(embeddings.embed_query(query))
        d_texts = [d.page_content for d in docs]
        d_vecs = [np.array(v) for v in embeddings.embed_documents(d_texts)]

        def cos(a, b):
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            return float(np.dot(a, b) / denom)

        # Compute scores and pair with docs
        scored_docs = []
        for doc, d_vec in zip(docs, d_vecs):
            score = cos(q_vec, d_vec)
            if score >= threshold:
                scored_docs.append((doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return only docs (no scores), up to top_k
        return [doc for doc, _ in scored_docs[:top_k]]

    return RunnableLambda(_rerank)


# ========================================
# Dictionary and spell correction
# ========================================
STOPWORDS = {
    "the","is","and","or","of","to","a","in","on","for","with","at",
    "are","you","we","it","do","have","what","how","i"
}
def build_dictionary():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found!")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = f.read().lower()

    words = re.findall(r"[a-zA-Z0-9]+", content)
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
    word_counts = Counter(filtered)

    with open(OUTPUT_DICT, "w", encoding="utf-8") as f:
        for word, count in word_counts.items():
            f.write(f"{word} {count}\n")

    print(f"✅ Dictionary created: {OUTPUT_DICT}")
    print(f"📌 Total unique words: {len(word_counts)}")

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
if os.path.exists(OUTPUT_DICT):
    sym_spell.load_dictionary(OUTPUT_DICT, term_index=0, count_index=1)
else:
    # safe load if not yet built; will rebuild on first query
    pass

def correct_spelling(text: str) -> str:
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# ========================================
# Intent detection
# ========================================
def find_brochure_file():
    if not os.path.exists(DATA_DIR):
        return None
    brochure_patterns = [
        r".*brochure.*\.(pdf|PDF)",
        r".*catalog.*\.(pdf|PDF)",
        r".*info.*\.(pdf|PDF)",
        r".*details.*\.(pdf|PDF)",
    ]
    for filename in os.listdir(DATA_DIR):
        for pattern in brochure_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return os.path.join(DATA_DIR, filename)
    return None

def detect_special_intent(query: str):
    q = query.lower().strip()
    if re.search(r"\b(brochure|brouchure|catalog|pdf|info sheet)\b", q):
        path = find_brochure_file()
        return {"type": "brochure", "content": path}
    meeting_text = "Call to Mr. akash on +91 8208333212 and Schedule a meeting.."
    if re.search(r"\b(schedule|book|set up|arrange).*?(call|meeting|appointment|demo)\b", q):
        return {"type": "schedule", "content": meeting_text}
    return None

def detect_society_in_query(q: str, known_names: set[str]) -> Optional[str]:
    m = NAME_RX.search(q)
    if m:
        return (" ".join([m.group(1), m.group(2)])).strip().title()
    q_low = q.lower()
    for name in sorted(known_names, key=len, reverse=True):
        if name.lower() in q_low:
            return name
    return None

def majority_society(docs) -> Optional[str]:
    vals = [d.metadata.get("society","unknown") for d in docs if getattr(d, "metadata", None)]
    if not vals:
        return None
    c = Counter(vals)
    top, _ = c.most_common(1)[0]
    return None if top.lower() == "unknown" else top

class SocietyFilteredRetriever:
    def __init__(self, base_retriever, society: Optional[str], k=8):
        self.base = base_retriever
        self.society = society
        self.k = k

    def invoke(self, query: str):
        docs = self.base.invoke(query)
        if not self.society:
            return docs[: self.k]
        filtered = [d for d in docs if (d.metadata or {}).get("society","").lower() == self.society.lower()]
        if len(filtered) >= max(3, self.k // 2):
            return filtered[: self.k]
        need = self.k - len(filtered)
        extras = [d for d in docs if d not in filtered]
        return (filtered + extras[:need])[: self.k]

# ========================================
# Chat history: fetch, select, format
# ========================================


# ✅ Modified history fetcher (sync call allowed inside async)
def fetch_chat_history(title_id, max_pairs=4):  # ✅ set number of turns you want
    if not title_id:
        return []

    try:
        url = f"http://13.204.247.180:7601/chatbot/getAllChats?title_id={title_id}"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()

        chat_pairs = []
        temp_user_msg = None

        if "data" in data:
            for entry in data["data"]: 
                print(entry)
                sender = entry.get("sender", "").strip().lower()
                message = entry.get("message", "").strip()

                if not message:
                    continue

                if sender == "user":
                    # ✅ Start new pair if a user message arrives
                    temp_user_msg = message

                elif sender == "bot":
                    # ✅ If bot replies, pair with last user message
                    chat_pairs.append({
                        "user": temp_user_msg or "",
                        "response": message
                    })
                    temp_user_msg = None

        # ✅ Return last N conversation turns (not only one)
        return chat_pairs[-max_pairs:]

    except Exception as e:
        print(f"[history] fetch failed: {e}")
        return []

def fetch_chat_history_local(title_id): 
    try:
        url = f"http://localhost:4001/history"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()

        chat_pairs = []
        temp_user_msg = None

        if "data" in data:
            for entry in data["data"]: 
                sender = entry.get("sender", "").strip().lower()
                message = entry.get("message", "").strip()

                if not message:
                    continue

                if sender == "user":
                    # ✅ Start new pair if a user message arrives
                    temp_user_msg = message

                elif sender == "bot":
                    # ✅ If bot replies, pair with last user message
                    chat_pairs.append({
                        "user": temp_user_msg or "",
                        "response": message
                    })
                    temp_user_msg = None

        # ✅ Return last N conversation turns (not only one)
        return chat_pairs[-6:]

    except Exception as e:
        print(f"[history] fetch failed: {e}")
        return []


def select_relevant_history(
    query: str,
    history: List[Dict[str, str]],
    embeddings,
    k_similar: int = 4,
    last_n: int = 4,
    max_chars: int = 1800,
) -> List[Dict[str, str]]:
    """
    Keep a small working memory:
      - last_n most recent turns
      - top k_similar semantically close older turns
      - truncate to max_chars after formatting
    """
    if not history:
        return []

    # Separate recent and older
    recent = history[-last_n:] if last_n > 0 else []
    older = history[:-last_n] if last_n > 0 else history

    # Semantic pick from older
    picks = []
    if older and k_similar > 0:
        q_vec = embeddings.embed_query(query)
        def pair_text(p): return f"User: {p.get('user','')}\nAssistant: {p.get('response','')}"
        older_texts = [pair_text(p) for p in older]
        older_vecs = embeddings.embed_documents(older_texts)

        def cos(a, b):
            a = np.array(a); b = np.array(b)
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            return float(np.dot(a, b) / denom)

        scores = [cos(q_vec, v) for v in older_vecs]
        idx_sorted = np.argsort(scores)[::-1][:k_similar]
        picks = [older[i] for i in idx_sorted]

    # Combine while preserving order for recent; picks keep their relative order by original index
    combined = picks + recent

    # Trim by character budget
    def fmt(hist):
        lines = []
        for h in hist:
            u = h.get("user","").strip()
            a = h.get("response","").strip()
            if u:
                lines.append(f"User: {u}")
            if a:
                lines.append(f"Assistant: {a}")
            lines.append("")  # blank line
        return "\n".join(lines).strip()

    # If too long, drop oldest from picks first
    while combined and len(fmt(combined)) > max_chars:
        if picks:
            picks.pop(0)
            combined = picks + recent
        elif recent:
            recent.pop(0)
            combined = picks + recent
        else:
            break
    return combined

def format_history_for_prompt(hist: List[Dict[str, str]]) -> str:
    if not hist:
        return ""
    parts = []
    for h in hist:
        u = h.get("user","").strip()
        a = h.get("response","").strip()
        if u:
            parts.append(f"User: {u}")
        if a:
            parts.append(f"Assistant: {a}")
        parts.append("")
    return "\n".join(parts).strip()

# ================================================================================= 
#      Saving Context in context.txt 

def load_context_file():
    if not os.path.exists(CONTEXT_FILE):
        return []
    try:
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []

def append_context_to_file(new_pair: dict):
    context = load_context_file()
    context.append(new_pair)
    if len(context) > 2:
        context = context[-2:]
        # print(context)
    with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2) 


# ==================================================================================
# removing the context when new chat start 
LAST_TITLE_ID = None
def reset_context():
    global LAST_TITLE_ID
    LAST_TITLE_ID = None
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
            f.write("[]")
    SESSION_STATE.clear()
    print("✅ Context reset due to new title_id")

# ========================================
# Prompt and LLM
# ========================================
def create_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a polite, concise, highly accurate real estate assistant named Arya.\n"
         "Rules:\n"
         "• If the user greets (hi/hello/hey/good morning/etc.), respond with a short friendly greeting, then ask how you can help with real estate.\n"
         "• Otherwise, DO NOT greet repeatedly. Answer the query directly and move the conversation forward.\n"
         "• Use ONLY the information in Knowledge and the limited Chat History for continuity.\n"
         "• If Knowledge is empty or insufficient for the request, ask a SPECIFIC follow-up to gather exactly what’s missing (e.g., location, BHK, budget, possession timeline) — do not repeat the same generic question.\n"
         "• If the question is unrelated to real estate, reply: 'I can only help with real estate-related questions.'\n"
         "• Do not mention 'Knowledge', 'documents', or 'sources'.\n"
         "• For direct factual requests, answer directly using Knowledge.\n"
         "• Keep responses SHORT and CLEAR. Always respond in English.\n\n"
         "Active Society: {active_society}\n\n"
         "Chat History (selected):\n{chat_history}\n\n"
         "Knowledge:\n{context}"
        ),
        ("human", "{question}")
    ])


import re

def build_reference_set(text: str):
    """
    Extracts all unique numeric patterns and key property terms 
    from the given text for reference correction.
    """
    numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', text)
    words = re.findall(r'\b[A-Za-z]{3,}\b', text)
    reference_set = set(numbers + words)
    return reference_set

def smart_fix_spaces_dynamic(text: str, reference_words: set = None) -> str:
    # --- 1️⃣ Fix numeric spacing (numbers, decimals, ranges, times) ---
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)           # 1 0 → 10
    text = re.sub(r'(\d)\s*:\s*(\d)', r'\1:\2', text)      # 10 : 90 → 10:90
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)     # ₹1 . 5 → ₹1.5
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1–\2', text)      # 850 - 1100 → 850–1100 (en dash)
    text = re.sub(r'([₹])\s*([0-9])', r'\1\2', text)       # ₹ 95 → ₹95
    text = re.sub(r'(\d)(\s*)(lakhs|crores)', r'\1 \3', text, flags=re.IGNORECASE)

    # --- 2️⃣ Fix common real-estate patterns ---
    text = re.sub(r'(\d)\s*B\s*H\s*K', r'\1BHK', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d)\s*BHK', r'\1BHK', text, flags=re.IGNORECASE)
    text = re.sub(r'\bQ\s*([1-4])\s*([0-9]{4})\b', r'Q\1 \2', text)  # Q 42026 → Q4 2026
    text = re.sub(r'\b(\d+)\s*(sq\s*ft|SQ\s*FT|Sq\s*Ft)\b', r'\1 sq ft', text, flags=re.IGNORECASE)

    # --- 3️⃣ Fix names like "Mr. Ak ash" ---
    text = re.sub(r'\b(Mr|Ms|Mrs|Dr)\.\s+([A-Z])\s+([a-z]+)', r'\1. \2\3', text)

    # --- 4️⃣ Fix URL / domain spacing ---
    text = re.sub(r'https\s*:\s*/\s*/\s*', 'https://', text)
    text = re.sub(r'www\s*\.\s*', 'www.', text)
    text = re.sub(r'\s*\.\s*com', '.com', text)
    text = re.sub(r'\s*\.\s*in', '.in', text)
    text = re.sub(r'\s*\.\s*org', '.org', text)
    text = re.sub(r'\s*\.\s*net', '.net', text)

    # --- 5️⃣ Context-based joining (reliable only for long words) ---
    if reference_words:
        for word in sorted(reference_words, key=len, reverse=True):
            if len(word) >= 4:
                pattern = r'\b' + r'\s*'.join(list(word)) + r'\b'
                text = re.sub(pattern, word, text, flags=re.IGNORECASE)

    # --- 6️⃣ General punctuation & space cleanup ---
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # no space before punctuation
    text = re.sub(r'([.,!?;:])([A-Za-z0-9])', r'\1 \2', text)  # ensure space after punctuation
    text = re.sub(r'\s{2,}', ' ', text)  # collapse multiple spaces
    text = text.strip()

    return text


# ========================================================
# High hello structured 


def normalize_text(text: str) -> str:
    """Normalize text for better fuzzy matching"""
    text = text.lower().strip()
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # hiiii -> hi
    text = re.sub(r'[^a-z\s]', '', text)      # remove punctuation
    return text

def similarity(a: str, b: str) -> float:
    """Compute similarity ratio"""
    return SequenceMatcher(None, a, b).ratio()

def detect_greeting(text: str):
    """Return greeting type and whether it's a standalone greeting"""
    responses_map = {
        "hello": "Hello! How i can Help you?",
        "hi": "Hi there! How i can help you?",
        "hey": "Hey! I'm here",
        "greetings": "Greetings! How may I be of service?",
        "good morning": "Good morning! How can I help you today?",
        "good afternoon": "Good afternoon! What would you like to know?",
        "good evening": "Good evening! How may I assist you?",
        "good day": "Good day to you! How can I help?",
        "what's up": "Not much, just ready to help you! What do you need?",
        "how are you": "I'm doing well, thank you! How can I help you today?",
        "morning": "Good morning! How can I assist you today?",
        "afternoon": "Good afternoon! How may I help you?",
        "evening": "Good evening! What can I do for you?",
        "good night": "Good night! Is there anything I can help you with before you go?",
        "howdy": "Howdy! What can I help you with today?",
        "yo": "Hello! How may I assist you?",
        "hi there": "Hi there! How can I help you?",
        "hello there": "Hello there! What would you like to know?",
        "good to see you": "Good to see you too! How can I assist you today?",
        "nice to see you": "Nice to see you as well! What can I help you with?",
        "long time no see": "Hello! It's good to connect with you again. How may I help?",
        "what's new": "Nothing much here! How can I assist you today?",
        "how's it going": "Things are going well! How may I help you?",
        "how have you been": "I've been doing great, thank you for asking! How can I assist you?",
        "hope you're well": "Thank you for asking! I'm here and ready to help. What do you need?",
        "good to meet you": "Good to meet you too! How may I be of service?",
        "pleased to meet you": "Pleased to meet you as well! What can I help you with today?",
        "welcome": "Thank you! How can I assist you?",
        "hey there": "Hey there! I'm ready to help. What do you need?",
        "what's happening": "Not much on my end! How can I help you today?",
        "namaste": "Namaste! How may I be of service?",
       "thanks": "You're welcome! Happy to help! 😊",
    "thank you": "Thank you! I'm glad I could assist you!",
    "thank you so much": "You're very welcome! I'm here whenever you need me! 🌟",
    "thanks a lot": "My pleasure! Don't hesitate to ask if you need anything else!",
    "thanks a bunch": "Anytime! I'm always happy to help! 😄",
    "much appreciated": "Glad to be of service! What else can I do for you?",
    "appreciate it": "You're welcome! That's what I'm here for!",
    "appreciate your help": "It's my pleasure to assist you! 🤗",
    "nice": "Awesome! Is there anything specific you'd like to know?",
    "that's nice": "Great! How can I make your day better?",
    "very nice": "Thank you! I'm here to provide the best help possible! 🌈",
    "so nice": "You're so kind! What can I assist you with today?",
    "nice one": "Thanks! Ready for whatever you need next! 👍",
    "good job": "Thank you! I'm here to help you succeed! 🚀",
    "well done": "Much appreciated! How else can I assist you?",
    "excellent": "Thank you! I'm dedicated to giving you excellent service! ⭐",
    "awesome": "You're awesome too! What can I help you with?",
    "great": "Great to hear! How may I continue assisting you?",
    "fantastic": "Fantastic! I'm here to make your experience better! 😎",
    "amazing": "You're amazing too! What would you like to know?",
    "perfect": "Perfect! I'm here to provide perfect assistance! 💫",
    "wonderful": "Wonderful! How can I help make your day even better?",
    "brilliant": "Thank you! I'm here to provide brilliant support! ✨",
    "outstanding": "Much appreciated! I'm committed to outstanding service! 🌟",
    "impressive": "Thank you! I'm impressed by your kindness! 😊",
    "super": "Super! I'm here to supercharge your experience! ⚡",
    "cool": "Cool! What can I help you explore today?",
    "sweet": "Sweet! Ready to assist you with anything! 🍭",
    "lovely": "Lovely to interact with you too! How can I help?",
    "beautiful": "You're beautiful too! What can I do for you? 🌸",
    "marvelous": "Marvelous! I'm here to provide marvelous support!",
    "splendid": "Splendid! How may I be of service to you?",
    "terrific": "Terrific! I'm here to give you terrific assistance! 🎯",
    "fabulous": "Fabulous! What fabulous thing can I help you with? 💖",
    "stellar": "Stellar! I'm here to provide stellar support! 🌠",
    "phenomenal": "Phenomenal! How can I phenomenally assist you?",
    "remarkable": "Remarkable! I'm here to provide remarkable help!",
    "exceptional": "Exceptional! What exceptional service can I provide? 🏆",
    "incredible": "Incredible! I'm here to incredibly assist you!",
    "extraordinary": "Extraordinary! How can I extraordinarily help you?",
    "magnificent": "Magnificent! I'm here to provide magnificent support! 👑",
    "many thanks": "Many welcomes! I'm here for all your needs!",
    "thanks a million": "A million welcomes! Always here to help! 💫",
    "much obliged": "The pleasure is mine! How else can I assist?",
    "deeply grateful": "I'm deeply happy to help! What's next?",
    "eternally grateful": "I'm eternally here for you! How can I help?",
    "highly appreciated": "Highly glad to assist! What do you need? 🌟", 
    "sounds good": "Thanks for using Our platform.",
    # Quick appreciative responses
    "thx": "You're welcome! 😊",
    "ty": "Anytime! What's up?",
    "tysm": "My pleasure! How can I help? 🌈",
    "tyvm": "You're very welcome! Ready for more!",
    "nice job": "Thank you! Happy to be of service! 👍",
    "good stuff": "Thanks! I'm here with more good stuff!",
    "well played": "Thank you! Ready for the next round! 🎮",
    
    # Appreciative with enthusiasm
    "you're the best": "No, you're the best! How can I help? 🌟",
    "you rock": "You rock too! What can I do for you? 🎸",
    "you're amazing": "You're more amazing! How may I assist?",
    "you're awesome": "Coming from you, that means a lot! 😄",
    "you're great": "You're greater! What can I help with?",
    "you're wonderful": "You're wonderful too! How can I serve you?",
    
    # Grateful responses
    "grateful": "I'm grateful to help you! What's next?",
    "so grateful": "I'm so happy to assist! How can I help?",
    "very grateful": "I'm very glad to be of service! 🌟",
    "extremely grateful": "I'm extremely happy to help you!",
    
    # Bless you responses
    "bless you": "Thank you! Bless you too! How can I help?",
    "god bless you": "Thank you! How may I assist you today?",
    "bless your heart": "You're so kind! What can I do for you? 💖"
    }

    normalized = normalize_text(text)
    best_match = None
    best_score = 0.0

    for key_greeting, response_message in responses_map.items():
        score = similarity(normalized, key_greeting)
        if score > best_score:
            best_score = score
            best_match = (key_greeting, response_message)

    # Detect if greeting word exists in start of message
    greeting_found = any(key in normalized.split()[:3] for key in responses_map)

    # 1️⃣ If the message is *only* a greeting → respond immediately
    if len(normalized.split()) <= 3 and (best_score >= 0.6 or greeting_found):
        return {"is_greeting": True, "response": best_match[1]}

    # 2️⃣ If greeting present + other intent words → skip greeting
    if greeting_found or best_score >= 0.6:
        return {"is_greeting": False, "response": None}

    # 3️⃣ Not a greeting at all
    return {"is_greeting": False, "response": None}
 
#=========================================================
def create_llm():

    llm = Llama( 
        model_path='models/qwen2.5-3b-instruct-q4_k_m.gguf',
        # model_path='models\Llama-3.2-3B.Q4_K_M.gguf',
        n_ctx=2048,
        n_threads=8,
        n_batch=1024 , 
        verbose=False
    )
    return llm

embeddings = create_embeddings()
rerank = create_reranker(embeddings, top_k=5)
prompt = create_prompt()  
global_llm  = create_llm() 

parser = StrOutputParser()

# ========================================
# FastAPI
# ========================================
app = FastAPI(title="Real Estate Chatbot")

@app.api_route("/stream_info", methods=["GET", "POST"])
async def ask_chat(request: Request ,  body: dict = Body(None)):
    global LAST_TITLE_ID

    title_id = None
    query = None

    # If POST → get title_id + query from body
    if request.method == "POST":
        title_id = (body or {}).get("title_id")
        query = (body or {}).get("query")

        if title_id and title_id != LAST_TITLE_ID:
            reset_context()
            LAST_TITLE_ID = title_id 
            
    else:
        query = request.query_params.get("query")
        title_id = request.query_params.get("title_id")

    session_id = get_session_id(request)

    if not query or not query.strip():
        return JSONResponse({
            "answer": ""
        })
        
    # ✅ INSERT GREETING HANDLER HERE
    greeting_check = detect_greeting(query)
    if greeting_check["is_greeting"] and greeting_check["response"]:
        async def greeting_stream():
            for ch in greeting_check["response"]:
                yield ch
                await asyncio.sleep(0.05)
        return StreamingResponse(greeting_stream(), media_type="text/plain; charset=utf-8")

    if not greeting_check["is_greeting"] and greeting_check["response"] is None:
        for key in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "namaste", "greetings"]:
            if query.lower().startswith(key):
                query = query[len(key):].strip(",.! ").strip()
                break

    # Intent handling
    intent = detect_special_intent(query)
    if intent:
        if intent["type"] == "brochure":
            response_text = (
                f"{os.path.basename(intent['content'])}"
                if intent["content"]
                else "Sorry, I couldn't find a brochure right now."
            )
        elif intent["type"] == "schedule":
            response_text = f"Great! You can schedule a call here: {intent['content']}"
        async def token_response():
            for ch in response_text:
                yield ch
        return StreamingResponse(token_response(), media_type="text/plain")

    # Detect society and manage session
    mentioned = detect_society_in_query(query, KNOWN_SOCIETIES)
    if mentioned:
        SESSION_STATE[session_id]["active_society"] = mentioned
    if re.search(r"\b(reset|clear)\b.*\bsociety\b", query, re.I):
        SESSION_STATE[session_id]["active_society"] = None

    active_society = SESSION_STATE[session_id].get("active_society")

    # Retrieve relevant context
    retriever = SocietyFilteredRetriever(base_retriever, active_society, k=8)
    docs = retriever.invoke(query)
    ranked_docs = rerank.invoke({"docs": docs, "question": query})
    context = format_docs(ranked_docs) 
    
    if len(context) > 2000:
        context = context[:2000]
        

    if not context.strip():
        import random

        fallback_messages = [
            "I don't have that information right now.",
            "Sorry, I couldn't find details about that.",
            "Hmm, I don't know the answer to that yet.",
            "It looks like I don't have enough information on this.",
            "I'm still learning — this one isn't in my knowledge base yet.",
            "Unfortunately, I don’t have data on that at the moment."
        ]
        fallback_response = random.choice(fallback_messages)

        async def fallback_stream():
            for ch in fallback_response:
                # Yield each character exactly as is (no cleaning)
                yield ch
                await asyncio.sleep(0.005)  # natural delay for streaming effect

        # Explicitly set charset to UTF-8
        return StreamingResponse(
            fallback_stream(),
            media_type="text/plain; charset=utf-8"
        )


    # Load local and external chat history
    if title_id!=None  and title_id!= 0: 
        full_history = fetch_chat_history(title_id) 
    else: 
        full_history = ""
    local_context = load_context_file()
    combined_history = (full_history or []) + local_context
    local_context = load_context_file()  # keep as text

    selected_hist = select_relevant_history(
        query,
        combined_history,
        embeddings,
        k_similar=4,
        last_n=4,
        max_chars=2000,
    ) or []

    new_turn = {"user": query, "response": ""}
    selected_hist.append(new_turn)
    chat_history_text = format_history_for_prompt(selected_hist) 
    if len(chat_history_text)> 800: 
        chat_history_text = chat_history_text[-800:]
    final_context_to_file = (
    f"User Query: {query}\n\n"
    f"Knowledge Used:\n{context}\n"
    f"{'='*60}\n"
)
    
    if  len(context)>10: 
        append_context_to_file({"user": query, "response": context}) 
 

    # 🧩 Clean system message for DeepSeek model
    llm = global_llm

    system_prompt = (
    "You are Arya — a warm, polite, and expert real-estate assistant. "
    "Your single source of truth is the section called 'Knowledge'. "
    "Treat the Knowledge content as verified, up-to-date, and directly relevant to the user's query. "
    "If the Knowledge includes any details about the user’s question (e.g., price, area, project name, BHK type, developer), "
    "you must answer using that information directly and confidently. "
    "⚠️ Preserve all numbers *exactly as written in the Knowledge section*, including zeros and commas (e.g., 1000, 25000, 3.50). Never round, truncate, or reformat them. "
    "Do not ask for the project or developer again — use what is provided in Knowledge. "
    "Only if Knowledge is completely empty should you ask a follow-up. "
    "Don't Use Certainly, first Conversation in response. "
    "Your tone should be empathetic, natural, and professional — like a helpful real estate consultant. "
    "Avoid generic responses or repeating the user's query. "
    "Be concise, accurate, and factual."
)
    chatml_prompt = f"""
<|system|>
{system_prompt}
<|end|>
<|user|>
Active Society: {active_society or "None"}

Chat History:
{chat_history_text}

The following Knowledge is guaranteed to be relevant to this user's query — it has been carefully retrieved from verified real estate data. Use it to answer directly.

Knowledge:
{context}

User Query:
{query}
<|end|>
<|assistant|>
""" 

    # print("=✔🌴"*30)
    # print("Total prompt Length: ", len(chatml_prompt.split())) 
    # # print("Prompt : \n" , chatml_prompt)
    # print("=✔🌴"*30)

    ###############################
    complete_context_reference_set = chat_history_text +"\n" + context
    reference_words = build_reference_set(complete_context_reference_set)  
    # print("=✔🌴"*30)
    # print(reference_words)
    # print("=✔🌴"*30)
    ################################
    # === STREAMING RESPONSE ===
    async def stream_response():
        # print("\n\n=====================")
        # print(f"🧠 User Query: {query}")
        # print("=====================")
        # print(f"📘 Context (truncated):\n{context[:400]}...\n")

        response_text = ""
        prev_chunk = ""
        word_buffer = []
        start_time = time.time()
        print("🎈🎈Start generating response ......")
        for token in llm(
            chatml_prompt,
            max_tokens=400,  # shorter and safer
            temperature=0.25,
            top_p=0.85,
            repeat_penalty=1.05,
            stream=True,
            presence_penalty=0.3,
            stop=["<|end|>", "<|user|>", "<|system|>", "\n\n\n"],
        ):
            chunk = token["choices"][0].get("text", "")
            if not chunk.strip():
                continue

            # Prevent duplication
            if chunk.strip() == prev_chunk.strip():
                continue
            prev_chunk = chunk

            # Skip unwanted HTML/control tokens
            if any(tag in chunk for tag in ["<div", "</", "<|user|>", "<|system|>"]):
                continue

            # Add missing space if needed
            if response_text and not response_text.endswith((" ", "\n")) and not chunk.startswith((" ", ".", ",", "!", "?")):
                chunk = " " + chunk

            # Collect into buffer
            words = chunk.split()
            word_buffer.extend(words)

            # 🧹 If buffer has 10 or more words, clean and yield
            if len(word_buffer) >= 10:
                segment = " ".join(word_buffer)
                cleaned_segment = smart_fix_spaces_dynamic(segment, reference_words=reference_words)

                print(cleaned_segment, end=" ", flush=True)
                yield cleaned_segment + " "

                response_text += cleaned_segment + " "
                word_buffer = []  # reset buffer

            await asyncio.sleep(0)

            # safety cutoff
            if len(response_text) > 1500:
                print("\n[🛑 Auto-stop after 1500 chars]\n")
                break

        # Flush remaining words
        if word_buffer:
            segment = " ".join(word_buffer)
            cleaned_segment = smart_fix_spaces_dynamic(segment, reference_words=reference_words)

            print(cleaned_segment, end=" ", flush=True)
            yield cleaned_segment + " "
            response_text += cleaned_segment + " "

        end_time = time.time()
        print("\n\n=====================")
        print("✅ Final Response:") 
        print(f"Total time  required to generate response: {end_time-start_time:.2f}")
        # print(response_text)
        # print("=====================\n")




    return StreamingResponse(
    stream_response(),
    media_type="text/plain",
    headers={"Transfer-Encoding": "chunked"}
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7602, log_level="info")
