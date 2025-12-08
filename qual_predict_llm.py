import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("Loading LLM sentiment summary data...")
summary = pd.read_csv("llm_summaries/sentiment_results.csv")  

print("Formatting summary dates (yy-mm-dd → yyyy-mm-dd)...")
summary["Date"] = pd.to_datetime(summary["date"], format="%y-%m-%d")

print("Loading stock price data...")
stocks = pd.read_csv("stocks/LMT_2010_2025.csv")

print("Converting stock columns to numeric...")
numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
for col in numeric_cols:
    stocks[col] = pd.to_numeric(stocks[col], errors="coerce")

stocks["Date"] = pd.to_datetime(stocks["Date"])
stocks = stocks.sort_values("Date").set_index("Date")

print("Computing price changes around each report date...")
price_changes = []

for _, row in summary.iterrows():
    report_date = row["Date"]

    if report_date in stocks.index:
        idx = stocks.index.get_loc(report_date)
        before_day = stocks.index[idx - 1] if idx > 0 else None
        after_day  = stocks.index[idx + 1] if idx < len(stocks.index)-1 else None
    else:
        future_dates = stocks.index[stocks.index >= report_date]

        if len(future_dates) < 2:
            continue  # insufficient data

        first_match = future_dates[0]
        pos = stocks.index.get_loc(first_match)

        before_day = stocks.index[pos - 1] if pos > 0 else None
        after_day  = stocks.index[pos + 1] if pos < len(stocks.index)-1 else None

    if before_day is None or after_day is None:
        continue

    before_price = stocks.loc[before_day]["Close"]
    after_price  = stocks.loc[after_day]["Close"]
    pct_change = (after_price - before_price) / before_price * 100

    price_changes.append({
        "Date": report_date,
        "avg_sentiment": row["avg_sentiment"],
        "pct_change": pct_change
    })

df = pd.DataFrame(price_changes)

if df.empty:
    print("No valid report and stock pairs found.")
    exit()


# ---------------------------
# Train/Test Split (80/20)
# ---------------------------
df = df.sort_values("Date")
split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[["avg_sentiment"]]
y_train = train["pct_change"]

X_test = test[["avg_sentiment"]]
y_test = test["pct_change"]

# ---------------------------
# Regression Model
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("Model coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# ----------------------------------------
# Visualization 3: Linear Regression Fit
# ----------------------------------------
plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train, alpha=0.7, label="Train Data")

# regression line
x_line = np.linspace(X_train.min(), X_train.max(), 200)
y_line = model.predict(x_line.reshape(-1,1))

plt.plot(x_line, y_line, linewidth=3, label="Regression Line")

plt.title("Linear Regression Fit (Training Data)")
plt.xlabel("Average Sentiment")
plt.ylabel("Percent Change")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Predict future values (test set)
# ---------------------------
test = test.copy()
test["predicted_change"] = model.predict(X_test)

# ---------------------------
# Accuracy Metrics
# ---------------------------
mae = mean_absolute_error(y_test, test["predicted_change"])
mse = mean_squared_error(y_test, test["predicted_change"])
r2  = r2_score(y_test, test["predicted_change"])

print("\n--- Prediction Accuracy ---")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²:  {r2:.4f}")

# ----------------------------------------
# Visualization 4: Actual vs Predicted
# ----------------------------------------
plt.figure(figsize=(10,6))
plt.plot(test["Date"], y_test, label="Actual Change", linewidth=2)
plt.plot(test["Date"], test["predicted_change"], label="Predicted Change", linewidth=2)
plt.title("Actual vs Predicted Stock Price Change (20% Future Data)")
plt.xlabel("Date")
plt.ylabel("Percent Price Change")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()