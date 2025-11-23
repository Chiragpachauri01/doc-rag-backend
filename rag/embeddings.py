import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyDd1j_Vve0u8ZHPpV2RzzGwk0CIEqiALbI")

def get_embedding(text: str):
    text = text.strip() or "empty"
    
    response = genai.embed_content(
        model="text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    
    return response["embedding"]
