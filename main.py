import os
from rag.qa import answer_question
from rag.text_cleaner import clean_text
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from rag.pdf_loader import extract_text_from_pdf
from rag.chunker import chunk_text
from rag.embeddings import get_embedding
from rag.vectorstore import create_collection_if_not_exists, add_chunks
from rag.qa import answer_question

load_dotenv()

app = FastAPI(title="Local RAG Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

create_collection_if_not_exists()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/check-key")
def check_key():
    return {"loaded": os.getenv("GEMINI_API_KEY")}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    uploads_dir = "data/uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = f"{uploads_dir}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 1️⃣ Extract PDF text
    text = extract_text_from_pdf(file_path)
    print("\n--- PDF TEXT LENGTH ---")
    print(len(text))

    # 2️⃣ Clean PDF text
    text = clean_text(text)
    print("\n--- CLEANED TEXT PREVIEW ---")
    print(text[:300])  # print first 300 chars

    # 3️⃣ Chunking
    chunks = chunk_text(text)
    print("\n--- NUMBER OF CHUNKS ---")
    print(len(chunks))

    if chunks:
        print("\n--- FIRST CHUNK PREVIEW ---")
        print(chunks[0][:300])  # show first chunk preview

    # 4️⃣ Embeddings
    payloads = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        print(f"Embedding {i} length:", len(emb))  # must be 768
        payloads.append({"text": chunk, "embedding": emb, "source": file.filename})

    # 5️⃣ Add to Qdrant
    add_chunks(payloads)

    return {"status": "ok", "file": file.filename, "chunks": len(chunks)}

@app.get("/models")
def list_models():
    import google.generativeai as genai
    return {"models": [m.name for m in genai.list_models()]}

@app.post("/ask")
async def ask(payload: dict):
    question = payload.get("question", "")
    return answer_question(question)
