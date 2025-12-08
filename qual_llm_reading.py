#This is the LLM method code. It generates the sentiment for the LLM method.

import ollama
import pandas as pd
import os
import re

model = "qwen3-coder"

os.makedirs("llm_summaries", exist_ok=True)

results = []

print("Scanning reports_formatted/ for report files...")
files = [f for f in os.listdir("reports_formatted") if f.startswith("lmt_") and f.endswith(".txt")]

for fname in files:
    print(f"Processing {fname}...")

    # extract date from filename: lmt_form_yy-mm-dd.txt
    date_part = fname.rsplit("_", 1)[1].replace(".txt", "")

    with open(f"reports_formatted/{fname}", encoding="utf-8") as txt_file:
        txt_read = txt_file.read()

    
    ai_summary = ""
    sentiments = []
    i = 1

    # chunk analysis
    while (i * 20000) < len(txt_read):
        txt_small = txt_read[(i - 1) * 20000 : i * 20000]

        question_prompt = (
            "Provide a sentiment score about this data from -1.0 (most negative) "
            "to 1.0 (most positive). Include no explanations. Respond only with the number."
        )

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": txt_small},
                {"role": "user", "content": question_prompt},
            ]
        )

        raw_answer = response["message"]["content"]

        # clean to extract only numeric value
        match = re.search(r"-?\d+\.\d+", raw_answer)
        if match:
            val = float(match.group())
            sentiments.append(val)
        else:
            print("Warning: AI returned no valid number.")
        
        i += 1

    # compute average sentiment
    avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0.0

    # store result
    results.append({
        "date": date_part,
        "avg_sentiment": avg_sent
    })

# save all results
df = pd.DataFrame(results)
df.to_csv("llm_summaries/sentiment_results.csv", index=False)

print("Saved llm_summaries/sentiment_results.csv")

