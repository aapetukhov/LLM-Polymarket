import json
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import os


NUM_CHUNKS = 18
BATCH_SIZE = 64
INPUT_DIR = "/kaggle/input/news-chunks-large"
OUTPUT_DIR = "./processed_chunks"
FULL_OUTPUT_PATH = "./full_dataset_ranked_deduplicated.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cuda')
model.max_length = 512


def prepare_pairs(entry):
    query = entry["title"] + " " + entry["description"]
    return [(query, article["text"]) for article in entry["articles"]]


def rank_articles(entry, scores):
    for article, score in zip(entry["articles"], scores):
        article["score"] = float(score)
    entry["articles"].sort(key=lambda x: x["score"], reverse=True)
    return entry


def remove_duplicate_articles(entry):
    seen_texts = set()
    unique_articles = []
    for article in entry.get("articles", []):
        text = article.get("text")
        if text not in seen_texts:
            seen_texts.add(text)
            unique_articles.append(article)
    entry["articles"] = unique_articles
    return entry


if __name__ == "__main__":
    all_results = []

    for i in range(NUM_CHUNKS):
        if i < 10:
            path = f"{INPUT_DIR}/news_chunk_0{i}.json"
            output_path = f"{OUTPUT_DIR}/news_chunk_0{i}_ranked.json"
        else:
            path = f"{INPUT_DIR}/news_chunk_{i}.json"
            output_path = f"{OUTPUT_DIR}/news_chunk_{i}_ranked.json"

        with open(path, "r", encoding="utf-8") as f:
            chunk = json.load(f)

        # ranking + cleaning
        processed_chunk = []
        for entry in tqdm(chunk, desc=f"Chunk {i}", leave=True):
            pairs = prepare_pairs(entry)
            scores = []
            for j in range(0, len(pairs), BATCH_SIZE):
                batch = pairs[j : j + BATCH_SIZE]
                batch_scores = model.predict(batch, show_progress_bar=False)
                scores.extend(batch_scores)
            entry = rank_articles(entry, scores)
            entry = remove_duplicate_articles(entry)
            processed_chunk.append(entry)
            all_results.append(entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_chunk, f, ensure_ascii=False, indent=2)

    with open(FULL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)