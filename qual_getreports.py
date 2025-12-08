#This code is used to clean the SEC pdf reports. It is not used in the analysis or results of the program. The data I've directed you to download in the readme is cleaned.

import os
import re
import pdfplumber
from datetime import datetime


input_folder = "reports"
output_folder = "reports_formatted"
os.makedirs(output_folder, exist_ok=True)

date_pattern = re.compile(r"Date:\s*(.+)", re.IGNORECASE)

def extract_text(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text.append(t)
    return "\n".join(text)

def extract_date_from_text(text, filename):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return None

    # --- Primary date extraction (last 50 lines) ---
    tail = lines[-50:] if len(lines) >= 50 else lines

    for line in reversed(tail):
        match = date_pattern.search(line)
        if match:
            raw_date = match.group(1).strip()

            cleaned = raw_date.split(",")
            if len(cleaned) >= 2:
                cleaned_date = cleaned[0].strip() + ", " + cleaned[1].strip()
            else:
                cleaned_date = raw_date

            try:
                dt = datetime.strptime(cleaned_date, "%B %d, %Y")
                return dt.strftime("%y-%m-%d")
            except Exception:
                pass

    # --- Failsafe 2: Extract year from filename ---
    try:
        middle_num = int(filename.split("-")[1])
    except Exception:
        return None

    if middle_num > 25:
        year_full = 1900 + middle_num
    else:
        year_full = 2000 + middle_num

    year_str = str(year_full)

    # Search last 200 lines for a line containing that year
    tail200 = lines[-200:] if len(lines) >= 200 else lines
    candidate_lines = [ln for ln in tail200 if year_str in ln]

    # Try each candidate to extract a full date
    for ln in candidate_lines:
        # Look for month-name dates on the same line
        date_match = re.search(
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*" + year_str,
            ln,
            re.IGNORECASE
        )
        if date_match:
            cleaned_date = date_match.group(0)
            try:
                dt = datetime.strptime(cleaned_date, "%B %d, %Y")
                return dt.strftime("%y-%m-%d")
            except Exception:
                pass

    # Nothing found
    return None


for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_folder, filename)
    print(f"Processing {filename}...")

    try:
        text = extract_text(pdf_path)
    except Exception as e:
        print(f"Failed to extract text from {filename}: {e}")
        continue

    date_str = extract_date_from_text(text, filename)
    if not date_str:
        print(f"Could not find date in {filename}; skipping.")
        continue

    out_name = f"lmt_report_{date_str}.txt"
    out_path = os.path.join(output_folder, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved → {out_name}")

print("Done.")

