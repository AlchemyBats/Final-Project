import pdfplumber

with pdfplumber.open("reports/10q25a.pdf") as pdf, open("reports_formatted/output.txt", "w", encoding="utf-8") as f:
    
    for page in pdf.pages:
        t = page.extract_text()
        if t:
            f.write(t + '\n')