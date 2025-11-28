# rag/qa.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

from rag.embeddings import get_embedding
from rag.vectorstore import search_user_chunks  # <-- NEW FILTERED SEARCH

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def answer_question(question: str, user: str, top_k: int = 5):
    """
    Full RAG pipeline with per-user filtering.
    'user' must be a unique string like 'anon_<uuid>' or the user's email.
    """

    # 1) Build embedding for the query
    query_emb = get_embedding(question)

    # 2) Search ONLY this user's chunks (THIS FIXES YOUR ISSUE)
    results = search_user_chunks(query_emb, user_id=user, top_k=top_k)

    if not results:
        return "No relevant information found.", []

    # 3) Build context for LLM
    context = ""
    sources = set()

    for r in results:
        text = r.payload.get("text", "")
        src = r.payload.get("source")
        context += f"[{src}]\n{text}\n\n"
        sources.add(src)

    # 4) Build prompt
    prompt = f"""
You are a helpful assistant.
Answer the user's question ONLY using the context below.

Context:
{context}

Question: {question}

If the answer is not found in the context, say:
"I could not find the answer in your documents."
"""

    # 5) Generate answer with Gemini
    model = genai.GenerativeModel("models/gemini-flash-latest")
    response = model.generate_content(prompt)

    return response.text, list(sources)
