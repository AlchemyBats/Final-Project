import os
import csv
import pandas as pd

# ---------------------------------------
# Load keyword groups from vector_keywords.csv
# ---------------------------------------
vec_df = pd.read_csv("vector_keywords.csv")

positive_keywords = vec_df["positive_keywords"].dropna().astype(str).str.lower().tolist()
neutral_keywords  = vec_df["neutral_keywords"].dropna().astype(str).str.lower().tolist()
negative_keywords = vec_df["negative_keywords"].dropna().astype(str).str.lower().tolist()

KEYWORDS = {
    "positive": positive_keywords,
    "neutral": neutral_keywords,
    "negative": negative_keywords
}

# ---------------------------------------
# Ensure output folder exists
# ---------------------------------------
os.makedirs("vector_summaries", exist_ok=True)

input_folder = "reports_formatted"
output_csv = "vector_summaries/results.csv"

results = []

# ---------------------------------------
# Process each report file
# ---------------------------------------
for fname in os.listdir(input_folder):
    if not fname.lower().startswith("lmt_") or not fname.lower().endswith(".txt"):
        continue

    # extract date: lmt_form_yy-mm-dd.txt
    try:
        date_part = fname.rsplit("_", 1)[1].replace(".txt", "")
    except Exception:
        continue

    # read file
    try:
        with open(os.path.join(input_folder, fname), "r", encoding="utf-8") as file:
            raw = file.read().lower()
    except Exception:
        continue

    # count keyword occurrences
    counts = {key: 0 for key in KEYWORDS}

    for category, words in KEYWORDS.items():
        for w in words:
            if len(w) > 1:  # avoid counting empty strings
                counts[category] += raw.count(w)

    # store
    results.append([
        date_part,
        counts["positive"],
        counts["negative"],
        counts["neutral"]
    ])

# ---------------------------------------
# Save results
# ---------------------------------------
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Date", "positives", "negatives", "neutral"])
    writer.writerows(results)

print("Saved:", output_csv)
