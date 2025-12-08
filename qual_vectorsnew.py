import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------
# 1. Load pretrained embeddings
# -------------------------------------------------------
print("Loading pretrained Word2Vec model...")
model_path = "pretrained/GoogleNews-vectors-negative300.bin"

wv = KeyedVectors.load_word2vec_format(model_path, binary=True)
vocab = set(wv.index_to_key)
print(f"Loaded pretrained model with {len(vocab)} tokens.")

# -------------------------------------------------------
# 2. Sentiment anchors
# -------------------------------------------------------
positive_seeds = ["growth", "profit", "strong", "improve", "increase", "gain"]
negative_seeds = ["loss", "risk", "decline", "weaken", "decrease", "drop"]

positive_seeds = [w for w in positive_seeds if w in vocab]
negative_seeds = [w for w in negative_seeds if w in vocab]

pos_centroid = np.mean([wv[w] for w in positive_seeds], axis=0)
neg_centroid = np.mean([wv[w] for w in negative_seeds], axis=0)

def score_word(word):
    v = wv[word]
    pos = np.dot(v, pos_centroid) / (np.linalg.norm(v) * np.linalg.norm(pos_centroid))
    neg = np.dot(v, neg_centroid) / (np.linalg.norm(v) * np.linalg.norm(neg_centroid))
    return pos - neg

# -------------------------------------------------------
# 3. Compute sentiment for each report
# -------------------------------------------------------
results = []

print("Computing sentiment for each report...")

for file in sorted(os.listdir("reports_formatted")):
    if not file.endswith(".txt"):
        continue

    # file is like "lmt_report_17-02-09.txt"
    # extract only the date portion
    date_str = file.replace("lmt_report_", "").replace(".txt", "")

    with open(os.path.join("reports_formatted", file), "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().lower()

    tokens = [w for w in word_tokenize(text) if w.isalpha() and w in vocab]

    if len(tokens) == 0:
        avg_sent = 0.0
    else:
        scores = [score_word(w) for w in tokens]
        avg_sent = float(np.mean(scores))*100

    results.append({"date": date_str, "avg_sentiment": avg_sent})
    print(f"{file}: sentiment = {avg_sent:.4f}")

# -------------------------------------------------------
# 4. Save results EXACTLY in required format
# -------------------------------------------------------
os.makedirs("vector_summaries", exist_ok=True)

df = pd.DataFrame(results, columns=["date", "avg_sentiment"])

df.to_csv("vector_summaries/results.csv", index=False)

print("Saved vector_summaries/results.csv")
