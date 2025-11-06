import os
import json
import re
import warnings
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter, defaultdict

import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
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
import httpx

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


SESSION_STATE = defaultdict(dict)  # per-session state: {"active_society": str}

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

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
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
def create_reranker(embeddings, top_k: int = 3) -> RunnableLambda:
    def _rerank(inputs):
        docs = inputs["docs"]
        query = inputs["question"]
        if not docs:
            return []

        q_vec = embeddings.embed_query(query)
        d_texts = [d.page_content for d in docs]
        d_vecs = embeddings.embed_documents(d_texts)

        def cos(a, b):
            a = np.array(a)
            b = np.array(b)
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            return float(np.dot(a, b) / denom)

        scores = [cos(q_vec, dv) for dv in d_vecs]
        ranked = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        return ranked[:top_k]
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
        url = f"http://15.206.70.213:7601/chatbot/getAllChats?title_id={title_id}"
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
    with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except:
            return []

def save_context_file(history: List[Dict[str, str]]):
    trimmed = history[-MAX_CONTEXT_TURNS:]  # Keep last 4
    with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2)

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
         "• Use ONLY the information in Knowledge and the limited Chat History for continuity.\n"
         "• If Knowledge is empty or insufficient, say: 'Sorry, I don’t have that information.'\n"
         "• If the question is unrelated to real estate, reply: 'I can only help with real estate-related questions.'\n"
         "• Do not mention 'Knowledge', 'documents', or 'sources'.\n"
         "• For direct factual requests, answer directly.\n"
         "• Ask clarifying questions for vague requests.\n"
         "• English only.\n\n"
         "Active Society: {active_society}\n\n"
         "Chat History (selected):\n{chat_history}\n\n"
         "Knowledge:\n{context}"
        ),
        ("human", "{question}")
    ])

def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.15,
        max_tokens=512,
        streaming=True,
    )

embeddings = create_embeddings()
rerank = create_reranker(embeddings, top_k=3)
prompt = create_prompt()  
llm = create_llm()
parser = StrOutputParser()

# ========================================
# FastAPI
# ========================================
app = FastAPI(title="Real Estate Chatbot")

@app.api_route("/stream_info", methods=["GET", "POST"])
async def ask_chat(request: Request):
    if request.method == "POST":
        form = await request.form()
        query = form.get("query")
        title_id = form.get("title_id")  # ✅ extract title ID 
        
        global LAST_TITLE_ID
        if title_id and title_id != LAST_TITLE_ID:
            reset_context()
            LAST_TITLE_ID = title_id

    else:
        query = request.query_params.get("query")
        title_id = request.query_params.get("title_id")


    session_id = get_session_id(request)

    if not query or not query.strip():
        welcome_text = "Welcome sir! How can I assist you regarding real estate queries?"
        async def stream_welcome():
            for ch in welcome_text:
                yield ch
        return StreamingResponse(stream_welcome(), media_type="text/plain")

    # Build spelling dictionary if needed
    if not os.path.exists(OUTPUT_DICT):
        build_dictionary()
        sym_spell.load_dictionary(OUTPUT_DICT, term_index=0, count_index=1)

    # Intent short-circuit
    intent = detect_special_intent(query)
    if intent:
        if intent["type"] == "brochure":
            path = intent["content"]
            response_text = f"{os.path.basename(path)}" if path else "Sorry, I couldn't find a brochure at the moment."
        elif intent["type"] == "schedule":
            response_text = f"Great! You can schedule a call with us here: {intent['content']}"
        async def token_response():
            for ch in response_text:
                yield ch
        return StreamingResponse(token_response(), media_type="text/plain")

    # Society state update
    mentioned = detect_society_in_query(query, KNOWN_SOCIETIES)
    if mentioned:
        SESSION_STATE[session_id]["active_society"] = mentioned
    if re.search(r"\b(reset|clear)\b.*\bsociety\b", query, re.I):
        SESSION_STATE[session_id]["active_society"] = None

    active_society = SESSION_STATE[session_id].get("active_society")

    # Build a society-filtered retriever
    retriever = SocietyFilteredRetriever(base_retriever, active_society, k=8)
    docs = retriever.invoke(query)

    # Auto-lock society by majority if none set yet
    if not active_society:
        maj = majority_society(docs)
        if maj:
            SESSION_STATE[session_id]["active_society"] = maj
            active_society = maj

    # Rerank and format knowledge
    ranked_docs = rerank.invoke({"docs": docs, "question": query})
    context = format_docs(ranked_docs)

    # Fetch and select chat history
    # 1) Online history retrieval
    full_history = fetch_chat_history(title_id)
    print(f"[history] entries fetched: {len(full_history)}")
    print("=="*30)
    print(full_history)
    print("=="*30)

    # 2) Load local context fallback if API fails or history is empty
    local_context = load_context_file()

    combined_history_for_selection = []
    if full_history:
        combined_history_for_selection += full_history
    combined_history_for_selection += local_context

    selected_hist = select_relevant_history(
        query,
        combined_history_for_selection,
        embeddings,
        k_similar=4,
        last_n=4,
        max_chars=1800
    )
    
    # Update context store for future requests
    if selected_hist is None:
        selected_hist = []
    new_turn = {"user": query, "response": ""}
    selected_hist.append(new_turn)
    save_context_file(selected_hist)

    async def token_response():
        full_answer = ""
        inputs = {
            "active_society": active_society or "None",
            "chat_history": chat_history_text,
            "context": context,
            "question": query,
        }
        async for chunk in chain.astream(inputs):
            full_answer += chunk
            yield chunk

        # ✅ Update the last turn with actual response
        selected_hist[-1]["response"] = full_answer
        save_context_file(selected_hist)



    chat_history_text = format_history_for_prompt(selected_hist)


    # Compose and stream LLM output
    chain = prompt | llm | parser

    async def get_full_response():
        full_answer = ""
        inputs = {
            "active_society": active_society or "None",
            "chat_history": chat_history_text,
            "context": context,
            "question": query,
        }
        async for chunk in chain.astream(inputs):
            full_answer += chunk
        return full_answer

    final_answer = await get_full_response()

    # ✅ Update context with full answer
    selected_hist[-1]["response"] = final_answer
    save_context_file(selected_hist)

    return JSONResponse({"response": final_answer})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7602, log_level="info")
