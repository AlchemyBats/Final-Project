import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


#Modifications
# - run the analysis on only 80% of the data
# - make a prediction about the price changes on each 'future' date based on the sentiment from the remaining 20%
# - compare predicted price changes to actual price changes (show graph)
# - print analysis results ranking the accuracy of the model to predicting price changes
print("Loading LLM sentiment summary data...")
summary = pd.read_csv("llm_summaries/sentiment_results.csv")  
# expected columns: Date, avg_sentiment

print("Formatting summary dates (yy-mm-dd → yyyy-mm-dd)...")
summary["Date"] = pd.to_datetime(summary["date"], format="%y-%m-%d")

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

    if report_date in stocks.index:
        idx = stocks.index.get_loc(report_date)
        before_day = stocks.index[idx - 1] if idx > 0 else None
        after_day  = stocks.index[idx + 1] if idx < len(stocks.index)-1 else None
    else:
        after_day = stocks.index[stocks.index.get_indexer([report_date], method="backfill")[0]]
        pos = stocks.index.get_loc(after_day)
        before_day = stocks.index[pos - 1] if pos > 0 else None
        after_day  = stocks.index[pos + 1] if pos < len(stocks.index)-1 else None

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
        "avg_sentiment": row["avg_sentiment"],
        "pct_change": pct_change
    })

df = pd.DataFrame(price_changes)

if df.empty:
    print("No valid report and stock pairs found.")
    exit()

print("Preparing regression model using avg_sentiment...")
X = df[["avg_sentiment"]]  
y = df["pct_change"]

model = LinearRegression()
model.fit(X, y)

print("Regression coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

print("Generating relationship chart...")

plt.figure(figsize=(8, 6))
plt.scatter(df["avg_sentiment"], df["pct_change"], s=80)

# Line of best fit
x_vals = np.linspace(df["avg_sentiment"].min(), df["avg_sentiment"].max(), 100).reshape(-1, 1)
y_vals = model.predict(x_vals)
plt.plot(x_vals, y_vals)

plt.title("Relationship Between LLM Avg Sentiment and Stock Price Change")
plt.xlabel("Avg Sentiment (LLM-derived)")
plt.ylabel("Percent Price Change After Filing")
plt.grid(True)

plt.tight_layout()
plt.show()
