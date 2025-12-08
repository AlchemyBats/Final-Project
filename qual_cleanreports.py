import os
import re
from datetime import datetime

# optional: dateutil gives more tolerant parsing for odd formats
try:
    from dateutil import parser as dateparser
except Exception:
    dateparser = None

os.makedirs("reports_formatted", exist_ok=True)

ticker = "LMT"
forms = ["10-K", "10-Q"]
base_dir = f"sec-edgar-filings/{ticker}"

def clean_text(text: str) -> str:
    # keep printable ASCII, newline, and common punctuation
    return re.sub(r"[^ \n\r\t\w\.\,\:\;\!\?\(\)\$\%\-\/]", "", text)

# regex patterns to try (ordered by likelihood)
DATE_PATTERNS = [
    r"(\d{4}-\d{2}-\d{2})",               # 2024-03-01
    r"(\d{2}/\d{2}/\d{4})",               # 03/01/2024
    r"(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", # 1 March 2024 or 01 March 2024
    r"([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})" # March 1, 2024
]

def find_date_in_snippet(snippet: str):
    # 1) try explicit regex patterns
    for pat in DATE_PATTERNS:
        m = re.search(pat, snippet)
        if m:
            s = m.group(1)
            # normalize obvious formats
            try:
                if re.match(r"\d{4}-\d{2}-\d{2}", s):
                    return datetime.strptime(s, "%Y-%m-%d")
                if re.match(r"\d{2}/\d{2}/\d{4}", s):
                    return datetime.strptime(s, "%m/%d/%Y")
                # try parsing month-name formats
                for fmt in ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y"):
                    try:
                        return datetime.strptime(s, fmt)
                    except Exception:
                        pass
            except Exception:
                pass

    # 2) try dateutil if available
    if dateparser:
        try:
            dt = dateparser.parse(snippet, fuzzy=True)
            if dt:
                return dt
        except Exception:
            pass

    # 3) last-resort: search for any 8+ digit sequence like YYYYMMDD
    m = re.search(r"(\d{8})", snippet)
    if m:
        s = m.group(1)
        try:
            return datetime.strptime(s, "%Y%m%d")
        except Exception:
            pass

    return None

for form in forms:
    form_dir = os.path.join(base_dir, form)
    if not os.path.isdir(form_dir):
        continue

    for filing_folder in os.listdir(form_dir):
        filing_path = os.path.join(form_dir, filing_folder)
        if not os.path.isdir(filing_path):
            continue

        # find primary file (.txt preferred, then .html)
        doc_file = None
        for ext in (".txt", ".html", ".htm"):
            for fname in os.listdir(filing_path):
                if fname.lower().endswith(ext):
                    doc_file = os.path.join(filing_path, fname)
                    break
            if doc_file:
                break
        if not doc_file:
            continue

        with open(doc_file, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        # find the first occurrence of ": " and take the following text
        idx = raw.find(": ")
        if idx == -1:
            # optional: try searching for 'Filed' or 'Filing Date' keywords as fallback
            fallback_idx = None
            for key in ("Filed", "Filing Date", "FilingDate", "filed"):
                kidx = raw.find(key)
                if kidx != -1:
                    # take substring after the key's first colon if present
                    colon = raw.find(":", kidx)
                    if colon != -1:
                        fallback_idx = colon
                        break
            if fallback_idx is None:
                # cannot locate date anchor; skip this filing
                print(f"Skipping (no ': ' or fallback) in {filing_path}")
                continue
            idx = fallback_idx

        snippet = raw[idx + 2 : idx + 2 + 400]  # grab a chunk after the first ": "
        # remove tags if HTML
        snippet = re.sub(r"<[^>]+>", " ", snippet)
        snippet = snippet.replace("\n", " ").strip()

        dt = find_date_in_snippet(snippet)
        if not dt:
            print(f"Could not parse date for filing {filing_folder}; using folder name fallback.")
            # fallback: try to get any YYYY in folder name and default to 01-01 if needed
            m = re.search(r"(20\d{2})", filing_folder)
            if m:
                year = int(m.group(1))
                dt = datetime(year, 1, 1)
            else:
                # final fallback: skip
                print(f"Skipping {filing_folder} (no parsable date).")
                continue

        date_str = dt.strftime("%y-%m-%d")

        cleaned = clean_text(raw)

        out_name = f"lmt_{form.lower()}_{date_str}.txt"
        out_path = os.path.join("reports_formatted", out_name)

        # ensure unique filename if multiple docs share same date+type
        counter = 1
        base_out = out_path
        while os.path.exists(out_path):
            out_path = base_out.replace(".txt", f"_{counter}.txt")
            counter += 1

        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(cleaned)

        print(f"Saved {out_path}")

print("Done.")
