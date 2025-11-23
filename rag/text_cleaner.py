import re

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)       # Remove excessive spaces
    text = re.sub(r'-\s+', '', text)       # Fix broken words due to hyphens
    text = re.sub(r'\n+', '\n', text)      # Normalize line breaks
    return text.strip()
