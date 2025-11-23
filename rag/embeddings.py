import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_embedding(text: str):
    text = text.strip() or "empty"
    
    response = genai.embed_content(
        model="text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    
    return response["embedding"]
