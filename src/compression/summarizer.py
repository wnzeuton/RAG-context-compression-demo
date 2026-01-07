from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_local(text):
    result = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return result[0]['summary_text']

def summarize_local_safe(text, chunk_size=500):
    # naive split by words
    words = text.split()
    summaries = []
    for i in range(0, len(words), chunk_size):
        sub_text = " ".join(words[i:i+chunk_size])
        summaries.append(summarize_local(sub_text))
    return " ".join(summaries)