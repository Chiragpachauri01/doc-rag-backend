import os
import google.generativeai as genai
from .embeddings import get_embedding
from .vectorstore import search_chunks
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def answer_question(question: str):
    query_emb = get_embedding(question)
    results = search_chunks(query_emb)

    if not results:
        return {"answer": "No relevant content found.", "sources": []}

    context = ""
    sources = set()

    for r in results:
        text = r.payload["text"]
        src = r.payload["source"]
        context += f"Source: {src}\n{text}\n\n"
        sources.add(src)

    prompt = f"""
You are a helpful assistant. Answer using ONLY the context below.

Context:
{context}

Question: {question}

Answer clearly. If not found, say: 'Answer not found in provided documents.'
"""

    model = genai.GenerativeModel("models/gemini-pro-latest")
    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": list(sources)
    }
