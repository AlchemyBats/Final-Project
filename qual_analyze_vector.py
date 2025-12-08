#goal: make predictions on stock changes after a 10k or 10q report is filed
#inputs:
#   manual_summaries/results.csv which includes the count of positive, negative, and neutral keywords in the reports
#   stocks/LMT_2024_2025.csv which includes all stock values from 2024-2025
#process:
#   given the changes in stock prices directly before and after a report is filed, and the number of positive/negative/neutral keywords in the report
#   use these values to create some sort of statistical regression relationship between these values
#   (note: use several print statements to explain the process to the user as the code runs)
#output:
#   a chart which highlights the relationship between keywords and price changes

#samples of inputs
#------
#manual_summaries/results.csv:
#   Date,positives,negatives,neutral
#   24-01-23,13300,49,4188
#   25-01-28,13562,63,4552
#stocks/LMT_2024_2025.csv:
#   Date,Close,High,Low,Open,Volume
#   ,LMT,LMT,LMT,LMT,LMT
#   2024-01-02,435.0343017578125,440.6901604150824,433.29843044333046,433.29843044333046,1206500
#   2024-01-03,437.8955993652344,442.64539227100187,436.64616093478446,437.5808675535479,1174300

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

print("Loading vector keyword summary data...")
summary = pd.read_csv("vector_summaries/results.csv")

print("Formatting summary dates (yy-mm-dd → yyyy-mm-dd)...")
summary["Date"] = pd.to_datetime(summary["Date"], format="%y-%m-%d")

print("Loading stock price data...")
stocks = pd.read_csv("stocks/LMT_2024_2025.csv")

print("Converting stock columns to numeric...")
numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
for col in numeric_cols:
    stocks[col] = pd.to_numeric(stocks[col], errors="coerce")

stocks["Date"] = pd.to_datetime(stocks["Date"])
stocks = stocks.sort_values("Date").set_index("Date")

print("Preparing to compute price changes for each report...")
price_changes = []

for _, row in summary.iterrows():
    report_date = row["Date"]

    print(f"Processing report filed on {report_date.date()}...")

    # find next trading day and previous trading day
    if report_date in stocks.index:
        before_day = stocks.index[stocks.index.get_loc(report_date) - 1] if stocks.index.get_loc(report_date) > 0 else None
        after_day  = stocks.index[stocks.index.get_loc(report_date) + 1] if stocks.index.get_loc(report_date) < len(stocks.index)-1 else None
    else:
        # find closest trading day after report
        after_day = stocks.index[stocks.index.get_indexer([report_date], method="backfill")[0]]
        pos = stocks.index.get_loc(after_day)
        before_day = stocks.index[pos-1] if pos > 0 else None
        after_day  = stocks.index[pos+1] if pos < len(stocks.index)-1 else None

    if before_day is None or after_day is None:
        print("Insufficient stock data around this date. Skipping.")
        continue

    before_price = stocks.loc[before_day]["Close"]
    after_price  = stocks.loc[after_day]["Close"]

    pct_change = (after_price - before_price) / before_price * 100
    print(f"  Price before: {before_price:.2f}")
    print(f"  Price after:  {after_price:.2f}")
    print(f"  Percent change: {pct_change:.3f}%")

    price_changes.append({
        "Date": report_date,
        "positives": row["positives"],
        "negatives": row["negatives"],
        "neutral": row["neutral"],
        "pct_change": pct_change
    })

df = pd.DataFrame(price_changes)

if df.empty:
    print("No valid report and stock pairs found.")
    exit()

print("Combining keyword counts into a single sentiment metric...")

# avoid division by zero
df["total_keywords"] = df["positives"] + df["negatives"] + df["neutral"]
df["sentiment"] = (df["positives"] - df["negatives"]) / df["total_keywords"]

print(df[["Date", "positives", "negatives", "neutral", "sentiment"]])

print("Preparing regression model using combined sentiment metric...")
X = df[["sentiment"]]   # single feature
y = df["pct_change"]

model = LinearRegression()
model.fit(X, y)

print("Regression coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

print("Generating relationship chart...")

plt.figure(figsize=(8, 6))
plt.scatter(df["sentiment"], df["pct_change"], s=80)

# Line of best fit
x_vals = np.linspace(df["sentiment"].min(), df["sentiment"].max(), 100).reshape(-1, 1)
y_vals = model.predict(x_vals)
plt.plot(x_vals, y_vals)

plt.title("Relationship Between Sentiment Score and Stock Price Change")
plt.xlabel("Sentiment Score (combined keywords)")
plt.ylabel("Percent Price Change After Filing")
plt.grid(True)

plt.tight_layout()
plt.show()

