import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# RAG imports
from rag.pdf_loader import extract_text_from_pdf
from rag.text_cleaner import clean_text
from rag.chunker import chunk_text
from rag.embeddings import get_embedding
from rag.vectorstore import create_collection_if_not_exists, add_chunks
from rag.qa import answer_question

# Auth imports
from auth_utils import (
    create_user,
    authenticate_user,
    create_access_token,
    decode_access_token,
    get_user,
)


# -----------------------------
# INITIAL SETUP
# -----------------------------
load_dotenv()
app = FastAPI(title="Local RAG Backend", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

create_collection_if_not_exists()


# -----------------------------
# AUTH MODELS
# -----------------------------
class RegisterIn(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = ""


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginIn(BaseModel):
    email: str
    password: str


# -----------------------------
# AUTH ENDPOINTS
# -----------------------------
@app.post("/auth/register", response_model=dict)
async def register(payload: RegisterIn):
    try:
        create_user(payload.email, payload.password, payload.full_name)
    except ValueError:
        raise HTTPException(status_code=400, detail="User already exists")

    return {"status": "ok", "email": payload.email}


@app.post("/auth/login", response_model=TokenOut)
async def login(payload: LoginIn):
    user = authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": payload.email})
    return {"access_token": token, "token_type": "bearer"}


# -----------------------------
# FIXED USER DETECTION
# supports:
# - Authorization: Bearer <token>
# - authorization: Bearer <token>   (Swagger lowercase)
# - X-Anonymous-ID: <uuid>
# -----------------------------
async def get_current_user_optional(
    authorization_upper: Optional[str] = Header(None, alias="Authorization"),
    authorization_lower: Optional[str] = Header(None, alias="authorization"),
    anon_id: Optional[str] = Header(None, alias="X-Anonymous-ID")
):
    # Prefer uppercase Authorization → then lowercase → then anonymous
    auth_header = authorization_upper or authorization_lower

    # CASE 1: Logged-in user (JWT)
    if auth_header:
        try:
            scheme, token = auth_header.split()
            if scheme.lower() == "bearer":
                payload = decode_access_token(token)
                if payload and payload.get("sub"):
                    email = payload["sub"]
                    user = get_user(email)
                    return {"email": email, "anonymous": False, **(user or {})}
        except Exception as e:
            print("JWT error:", e)

    # CASE 2: Anonymous user with UUID
    if anon_id:
        return {"email": f"anon_{anon_id}", "anonymous": True}

    # CASE 3: Unknown (should not happen normally)
    return {"email": "anon_unknown", "anonymous": True}


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# PER-USER UPLOAD
# -----------------------------
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    user_id = current_user["email"]

    uploads_dir = f"data/uploads/{user_id}"
    os.makedirs(uploads_dir, exist_ok=True)

    file_path = f"{uploads_dir}/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # RAG pipeline
    text = extract_text_from_pdf(file_path)
    text = clean_text(text)
    chunks = chunk_text(text)

    payloads = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        payloads.append({
            "text": chunk,
            "embedding": emb,
            "source": file.filename,
            "user": user_id
        })

    add_chunks(payloads)

    return {
        "status": "ok",
        "file": file.filename,
        "user": user_id,
        "chunks": len(chunks)
    }


# -----------------------------
# ASK RAG (per user)
# -----------------------------
@app.post("/ask")
async def ask(
    payload: dict,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    question = payload.get("question", "")
    user_id = current_user["email"]

    answer, sources = answer_question(question, user=user_id)

    return {
        "answer": answer,
        "sources": sources,
        "user": user_id
    }


# -----------------------------
# LIST USER FILES
# -----------------------------
@app.get("/files")
async def list_files(
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    user_id = current_user["email"]
    user_dir = f"data/uploads/{user_id}"
    os.makedirs(user_dir, exist_ok=True)

    files = os.listdir(user_dir)

    result = [
        {
            "id": f"{user_id}/{f}",
            "name": f,
            "snippet": "Document indexed and ready.",
            "tag": "General",
            "pages": 0,
            "updated": "Just now"
        }
        for f in files
    ]

    return {
        "user": user_id,
        "files": result
    }
