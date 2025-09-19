import os
import re
import sys
import json
import time
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
import tempfile

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import gradio as gr
from dotenv import load_dotenv
from ollama import Client, AsyncClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# =========================
# CONFIG
# =========================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, ".env"))

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(os.path.dirname(__file__), "vectorstore"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL   = os.getenv("LLM_MODEL", "llama3:8b")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "10"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_bundles")

# =========================
# OLLAMA CLIENTS
# =========================
ollama_sync = Client(host=OLLAMA_HOST)
ollama_async = AsyncClient(host=OLLAMA_HOST)

# =========================
# HELPERS
# =========================
def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def read_bundle_md_files(output_root: str) -> List[Dict[str, Any]]:
    bundles = []
    for bundle in Path(output_root).glob("*/bundle.md"):
        doc_id = bundle.parent.name
        text = bundle.read_text(encoding="utf-8")
        bundles.append({
            "doc_id": doc_id,
            "path": str(bundle),
            "text": text,
            "sha1": sha1_text(text),
        })
    return bundles

def detect_page_number_context(lines: List[str]) -> List[Tuple[int, str]]:
    results = []
    current_page = None
    page_pattern = re.compile(r"^##\s*Page\s+(\d+)\s*$")
    for line in lines:
        m = page_pattern.match(line.strip())
        if m:
            try:
                current_page = int(m.group(1))
            except:
                current_page = None
        results.append((current_page, line))
    return results

def chunk_text_preserve_pages(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    lines = text.split("\n\n")
    page_annotated = detect_page_number_context(lines)
    chunks = []
    buf = []
    cur_len = 0
    cur_page = None

    def flush_chunk():
        nonlocal buf, cur_len, cur_page
        if buf:
            chunk_text = "\n\n".join(buf).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": cur_page
                })
        buf = []
        cur_len = 0

    for page_no, block in page_annotated:
        block = block.strip()
        if not block:
            continue
        if cur_page is None:
            cur_page = page_no

        if cur_len + len(block) + 2 > chunk_size and buf:
            joined = "\n\n".join(buf)
            if overlap > 0 and len(joined) > overlap:
                keep_tail = joined[-overlap:]
                flush_chunk()
                buf = [keep_tail]
                cur_len = len(keep_tail)
            else:
                flush_chunk()

        buf.append(block)
        cur_len += len(block) + 2

        if block.startswith("## Page") and len(buf) > 1:
            buf.pop()
            cur_len -= len(block) + 2
            flush_chunk()
            cur_page = page_no
            buf = [block]
            cur_len = len(block)

    flush_chunk()
    return [c for c in chunks if c["text"].strip()]

def embed_texts_ollama(texts: List[str]) -> List[List[float]]:
    resp = ollama_sync.embed(model=EMBED_MODEL, input=texts)
    embs = resp.get("embeddings")
    if embs is None:
        embs = [r["embedding"] for r in resp["data"]]
    return embs

# =========================
# CHROMA SETUP
# =========================
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(allow_reset=False)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def index_bundle(doc_id: str, text: str, sha1: str, source_path: str):
    existing = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    needs_reindex = True
    if existing and existing.get("metadatas"):
        metas = existing["metadatas"]
        if metas and all(m.get("sha1") == sha1 for m in metas):
            needs_reindex = False

    if not needs_reindex:
        print(f"• Skipping (up-to-date): {doc_id}")
        return

    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])
        print(f"• Reindexing: {doc_id}")

    chunks = chunk_text_preserve_pages(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        print(f"• No chunks for {doc_id} — skipping.")
        return

    documents = [c["text"] for c in chunks]
    metadatas = []
    ids = []
    for i, c in enumerate(chunks):
        metadatas.append({
            "doc_id": doc_id,
            "sha1": sha1,
            "source": source_path,
            "page": c["page"]
        })
        ids.append(f"{doc_id}::chunk::{i}")

    all_embeddings = []
    BATCH = 64
    for start in range(0, len(documents), BATCH):
        embs = embed_texts_ollama(documents[start:start+BATCH])
        all_embeddings.extend(embs)

    collection.upsert(
        ids=ids,
        metadatas=metadatas,
        documents=documents,
        embeddings=all_embeddings
    )
    print(f"• Indexed {doc_id}: {len(documents)} chunks.")

def ensure_indexed():
    bundles = read_bundle_md_files(OUTPUT_ROOT)
    if not bundles:
        print(f"❌ No bundle.md files found in {OUTPUT_ROOT}.")
        sys.exit(1)
    print(f"Found {len(bundles)} documents to index.")
    for b in bundles:
        index_bundle(b["doc_id"], b["text"], b["sha1"], b["path"])

# =========================
# RETRIEVAL + ANSWERING
# =========================
def retrieve_context(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    q_emb = embed_texts_ollama([query])[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    if not res or not res.get("documents"):
        return hits
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None]*len(docs)])[0]
    for text, meta, dist in zip(docs, metas, dists):
        hits.append({
            "doc_id": meta.get("doc_id"),
            "page": meta.get("page"),
            "text": text,
            "distance": dist,
            "source": meta.get("source")
        })
    return hits

async def answer_with_context(question: str) -> str:
    hits = retrieve_context(question, top_k=TOP_K)
    if not hits:
        return "No relevant context found."

    MAX_CTX_CHARS = 10000
    ctx_lines, total = [], 0
    for i, h in enumerate(hits, start=1):
        chunk = f"[{i}] Doc: {h['doc_id']} | Page: {h['page']}\n{h['text']}\n"
        if total + len(chunk) > MAX_CTX_CHARS:
            break
        ctx_lines.append(chunk)
        total += len(chunk)
    context = "\n".join(ctx_lines)

    prompt = (
        "Use ONLY the provided context to answer the question. "
        "Cite sources like [1], [2] matching the chunks. "
        f"\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    resp = await ollama_async.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return resp["message"]["content"]

# =========================
# PDF EXPORT
# =========================
chat_history = []

def export_chat_to_pdf():
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 40
    c.setFont("Helvetica", 12)

    for i, (q, a) in enumerate(chat_history, start=1):
        c.drawString(30, y, f"Q{i}: {q}")
        y -= 20
        for line in a.split("\n"):
            c.drawString(50, y, line)
            y -= 15
            if y < 40:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = height - 40
        y -= 10
    c.save()
    return pdf_path

# =========================
# MAIN
# =========================
def main():
    ensure_indexed()

    async def on_message(message, history):
        answer = await answer_with_context(message)
        # store in our own history for PDF
        chat_history.append((message, answer))
        # return updated messages in "messages" format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history

    with gr.Blocks() as demo:
        gr.Markdown("# 📚 Multi-Document PDF Chatbot")
        chatbot = gr.Chatbot(type="messages")  # new messages format
        msg = gr.Textbox(placeholder="Ask something...")
        btn = gr.Button("📥 Download Chat as PDF")
        file_out = gr.File()

        msg.submit(on_message, inputs=[msg, chatbot], outputs=chatbot)
        btn.click(export_chat_to_pdf, outputs=file_out)

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
