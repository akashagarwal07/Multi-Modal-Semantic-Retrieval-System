import os
import requests
import concurrent.futures
import threading
import re

INPUT = "pdf_list.txt"
PDF_DIR = "pdfspdf_repo"
THREADS = 20   # you can push to 40+ if your network is good

os.makedirs(PDF_DIR, exist_ok=True)

print_lock = threading.Lock()


def sanitize_filename(name):
    """Remove illegal filename characters."""
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)[:200]


def fetch_with_optional_insecure(url, timeout=30):
    """
    Try a normal HTTPS request first.
    If SSL fails, retry once with verify=False.
    """
    try:
        return requests.get(url, timeout=timeout)
    except requests.exceptions.SSLError:
        # Fallback: insecure, but useful for broken certs
        return requests.get(url, timeout=timeout, verify=False)


def download_pdf(url):
    filename = url.split("/")[-1] or "file.pdf"
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    filename = sanitize_filename(filename)
    path = os.path.join(PDF_DIR, filename)

    if os.path.exists(path):
        return (filename, "already")

    try:
        r = fetch_with_optional_insecure(url, timeout=40)
        status = r.status_code
        ctype = (r.headers.get("content-type") or "").lower()

        # Read small chunk to test magic header
        content = r.content
        is_pdf_header = content.startswith(b"%PDF")

        if status == 200 and ("pdf" in ctype or is_pdf_header):
            with open(path, "wb") as f:
                f.write(content)
            return (filename, "success")
        else:
            return (filename, f"invalid (status={status}, type={ctype})")

    except Exception as e:
        return (filename, f"error: {e}")


# Load URLs
with open(INPUT, "r", encoding="utf-8") as f:
    links = [x.strip() for x in f.readlines() if x.strip()]

total = len(links)
print("\n==============================")
print("   MULTITHREADED DOWNLOADER")
print("==============================")
print(f"Total URLs: {total}")
print(f"Saving to: {PDF_DIR}\n")

success = 0
invalid = 0
error_count = 0
already = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
    futures = {executor.submit(download_pdf, url): url for url in links}

    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        filename, status = future.result()

        with print_lock:
            print(f"[{i+1}/{total}] {filename} → {status}")

        if status == "success":
            success += 1
        elif status == "already":
            already += 1
        elif status.startswith("invalid"):
            invalid += 1
        else:
            error_count += 1

print("\n==============================")
print("           SUMMARY")
print("==============================")
print(f"Total URLs:         {total}")
print(f"Downloaded:         {success}")
print(f"Already existed:    {already}")
print(f"Invalid/HTML/etc.:  {invalid}")
print(f"Errors (network):   {error_count}")
print(f"\n📁 PDFs saved in → {PDF_DIR}")
