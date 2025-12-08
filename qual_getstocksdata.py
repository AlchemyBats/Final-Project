#This downloads the stock data I used. It is not needed in the analysis or results.

import os
import yfinance as yf
import pandas as pd

# Create folder if it doesn't exist
os.makedirs("stocks", exist_ok=True)

# Download stock data for Lockheed Martin (LMT) from 2024–2025
data = yf.download("LMT", start="1995-01-01", end="2025-12-31")

# Basic cleaning: drop rows with missing values and reset index
cleaned = data.dropna().reset_index()

# Save to CSV
cleaned.to_csv("stocks/LMT_1995_2025.csv", index=False)

print("Data downloaded, cleaned, and saved to stocks/LMT_2010_2025.csv")

