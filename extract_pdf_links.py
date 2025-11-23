import json

INPUT = "semantic_scholar_results.jsonl"
OUTPUT = "pdf_list.txt"

pdfs = []

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)

        # priority 1: openAccessPdf
        if p.get("openAccessPdf"):
            pdfs.append(p["openAccessPdf"])
            continue

        # priority 2: arXiv
        arx = p.get("arxivId")
        if arx:
            pdfs.append(f"https://arxiv.org/pdf/{arx}.pdf")
            continue

with open(OUTPUT, "w", encoding="utf-8") as f:
    for link in pdfs:
        f.write(link + "\n")

print(f"Found {len(pdfs)} PDFs")
print(f"Saved → {OUTPUT}")
