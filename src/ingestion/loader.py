from pathlib import Path
from src.config import RAW_DIR

def load_documents():
    docs = []
    for file in Path(RAW_DIR).glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        docs.append({
            "id": file.stem,
            "text": text
        })
    return docs

docs = load_documents()
print(len(docs), docs[0]["id"])