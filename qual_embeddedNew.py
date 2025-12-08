#goal: create a sentiment value for each report through vector similarities
#modifications:
#   train on data from reports_formatted instead of the zipped file
#   create a list of 50 most positive, neutral, and negative keywords based on vectors
#   (note: the words 'positive', 'negative', or 'neutral' are likely not directly in the dataset, so some other metric of determining value of keywords is needed)
#   save keywords to csv file vector_keywords.csv


#Currently, the model is trained on the existing reports and generates a list of the 50 most pos, neg, and neutral keywords.
# Then those keywords are plugged into the same analysis as the manual keywords to count.

#However, I know this isn’t very effective because it is currently training and reading from the same documents.

#Planned improvements: My current plan is to instead train the model on a completely unrelated corpus.
# Then instead of providing a list of pos, neg, neutral keywords and then counting their occurances, I will have the program
# find the 50 most pos and neg words (perhaps by using faiss’s searching method, or another simpler method) and the sentiment score will be generated
# by how positive or negative they are, noting their closeness to positive and negative words.

#Output: this code should output a csv file vector_summaries/results.csv that tracks the date of the file and the overall sentiment score
#for instance
#date,avg_sentiment
#17-02-09,0.25227272727272726

#
import os
import pandas as pd
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt_tab')
import warnings
warnings.filterwarnings(action='ignore')

# ------------------------------------------
# 1. Load all text data from reports_formatted
# ------------------------------------------
print("Loading report text files...")
corpus = ""

for file in os.listdir("reports_formatted"):
    if file.endswith(".txt"):
        with open(os.path.join("reports_formatted", file), "r", encoding="utf-8", errors="ignore") as f:
            corpus += " " + f.read()

print("Reports loaded.")

# ------------------------------------------
# 2. Tokenize into sentences and words
# ------------------------------------------
print("Tokenizing...")
data = []
for sentence in sent_tokenize(corpus):
    tokens = [w.lower() for w in word_tokenize(sentence)]
    data.append(tokens)

# ------------------------------------------
# 3. Train Word2Vec model
# ------------------------------------------
print("Training Word2Vec...")
model = Word2Vec(
    data,
    vector_size=100,
    window=5,
    min_count=3
)

wv = model.wv
vocab = list(wv.index_to_key)

print(f"Vocabulary size: {len(vocab)} words")

# ------------------------------------------
# 4. Build sentiment axis using seed anchors
# ------------------------------------------
positive_seeds = ["growth", "profit", "strong", "improve"]
negative_seeds = ["loss", "risk", "decline", "weaken"]

positive_seeds = [w for w in positive_seeds if w in vocab]
negative_seeds = [w for w in negative_seeds if w in vocab]

pos_centroid = sum(wv[w] for w in positive_seeds) / len(positive_seeds)
neg_centroid = sum(wv[w] for w in negative_seeds) / len(negative_seeds)

# ------------------------------------------
# 5. Score every word along the sentiment axis
# ------------------------------------------
print("Scoring vocabulary...")

def score_word(word):
    return wv.cosine_similarities(wv[word], [pos_centroid])[0] - \
           wv.cosine_similarities(wv[word], [neg_centroid])[0]

scores = {word: score_word(word) for word in vocab}

# ------------------------------------------
# 6. Select top keyword groups
# ------------------------------------------
sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

top_positive = sorted_words[:50]
top_negative = sorted_words[-50:]
# neutral = words closest to 0
top_neutral = sorted(sorted_words, key=lambda x: abs(x[1]))[:50]

# ------------------------------------------
# 7. Save to CSV
# ------------------------------------------
df = pd.DataFrame({
    "positive_keywords": [w for w, s in top_positive],
    "positive_scores": [s for w, s in top_positive],
    "neutral_keywords": [w for w, s in top_neutral],
    "neutral_scores": [s for w, s in top_neutral],
    "negative_keywords": [w for w, s in reversed(top_negative)],
    "negative_scores": [s for w, s in reversed(top_negative)]
})

df.to_csv("vector_keywords.csv", index=False)
print("Saved vector_keywords.csv")
