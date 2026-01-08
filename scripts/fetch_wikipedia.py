from datasets import load_dataset
from pathlib import Path
import shutil

# Directory to save articles
RAW_DIR = Path("data/raw")

# Clear the directory if it exists
if RAW_DIR.exists() and RAW_DIR.is_dir():
    shutil.rmtree(RAW_DIR)

# Recreate the empty directory
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Load the dog-specific dataset
dataset = load_dataset("SparkleDark/Everything_about_dogs", split="train")

# Settings
articles_per_file = 300
file_index = 0
article_count = 0
current_file_text = []

for row in dataset:
    text = row.get("text", "").strip()  # get text and remove leading/trailing spaces
    if not text:  # skip empty articles
        continue

    current_file_text.append(text)
    article_count += 1

    # Write to file every `articles_per_file` articles
    if article_count % articles_per_file == 0:
        file_path = RAW_DIR / f"wiki_dogs_{file_index}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(current_file_text))  # separate articles with 2 newlines
        print(f"Saved {len(current_file_text)} articles to {file_path}")
        file_index += 1
        current_file_text = []  # reset for next file

# Write any remaining articles
if current_file_text:
    file_path = RAW_DIR / f"wiki_dogs_{file_index}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(current_file_text))
    print(f"Saved {len(current_file_text)} articles to {file_path}")

print(f"Processed {article_count} dog-related articles in total.")