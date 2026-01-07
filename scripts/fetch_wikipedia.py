from datasets import load_dataset
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train[:50]"
)

for i, row in enumerate(dataset):
    text = row["text"]
    file_path = RAW_DIR / f"wiki_{i}.txt"
    file_path.write_text(text, encoding="utf-8")

print(f"Saved {len(dataset)} Wikipedia articles.")