import os
import json
import faiss
import torch
import numpy as np
import random
import math
import asyncio
import httpx 
import fitz  # REQUIREMENT: pip install pymupdf
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import AutoModel, AutoProcessor
from PIL import Image
import io

# --- CONFIG ---
EMB_DIR = "outputs/embeddings"
IDX_DIR = "outputs/indexes"
PAPER_MAP_FILE = "outputs/papers.json"
PDF_REPO = "pdf_repo" 
MODEL_ID = "jinaai/jina-clip-v1"

# !!! KEYS !!!
GEMINI_API_KEY = "AIzaSyDxUiv845c4QDzHnqK6FqpnP6W8vrW0xA0" 

app = FastAPI(title="Research Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/pdfs", StaticFiles(directory=PDF_REPO), name="pdfs")

state = {
    "model": None, "processor": None, 
    "index_txt": None, "index_img": None, 
    "meta_txt": [], "meta_img": [], "papers": {}
}

@app.on_event("startup")
async def load_resources():
    print("--- STARTING SERVER ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["device"] = device
    state["model"] = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    state["processor"] = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    state["index_txt"] = faiss.read_index(f"{IDX_DIR}/faiss_text.index")
    state["index_img"] = faiss.read_index(f"{IDX_DIR}/faiss_image.index")

    with open(f"{EMB_DIR}/text_metadata.json", "r") as f: state["meta_txt"] = json.load(f)
    with open(f"{EMB_DIR}/image_metadata.json", "r") as f: state["meta_img"] = json.load(f)
    with open(PAPER_MAP_FILE, "r") as f: state["papers"] = json.load(f)
    print("✅ Server Ready!")

@app.get("/")
def serve_ui():
    return FileResponse("appui/index.html")

# --- 7.1 SMART TITLE EXTRACTION (CPU FRIENDLY) ---
def get_smart_title_from_pdf(pid):
    """
    Extracts the Title by finding the text with the LARGEST FONT SIZE on Page 1.
    """
    try:
        path = state["papers"][pid]["path"]
        filename = state["papers"][pid]["filename"]

        # 1. Regex: Check for ArXiv ID in filename
        arxiv_match = re.search(r'(\d{4}\.\d{4,5})', filename)
        if arxiv_match:
            return f"arXiv:{arxiv_match.group(1)}"

        # 2. Heuristic: Open PDF and find largest text
        doc = fitz.open(path)
        page = doc[0] # Page 1
        blocks = page.get_text("dict")["blocks"]
        
        max_size = 0
        title_parts = []
        
        # Pass 1: Find the largest font size on the page
        for b in blocks:
            if "lines" not in b: continue
            for line in b["lines"]:
                for span in line["spans"]:
                    if span["size"] > max_size:
                        max_size = span["size"]
        
        # Pass 2: Collect all text that uses that max size (handling multi-line titles)
        # We allow a small tolerance (e.g. 0.5) because sometimes fonts vary slightly
        if max_size > 0:
            for b in blocks:
                if "lines" not in b: continue
                for line in b["lines"]:
                    for span in line["spans"]:
                        if span["size"] > (max_size - 1): # Tolerance
                            title_parts.append(span["text"].strip())
            
            extracted_title = " ".join(title_parts)
            # Remove weird characters if any
            clean_title = re.sub(r'\s+', ' ', extracted_title).strip()
            
            if len(clean_title) > 5: # Ensure we didn't just get a number
                return clean_title

    except Exception as e:
        pass
    
    # 3. Fallback: Filename cleanup
    return state["papers"][pid]["filename"].replace(".pdf", "").replace("_", " ")

async def fetch_citation_single(client, pid):
    try:
        # Get the REAL title from the PDF content
        search_query = get_smart_title_from_pdf(pid)
        
        # Semantic Scholar API Search
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={search_query}&limit=1&fields=citationCount"
        
        resp = await client.get(url, timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                return pid, data["data"][0].get("citationCount", 0)
    except Exception:
        pass
    return pid, 0 

async def fetch_real_citations_batch(candidates):
    async with httpx.AsyncClient() as client:
        tasks = []
        for pid in candidates.keys():
            tasks.append(fetch_citation_single(client, pid))
        results = await asyncio.gather(*tasks)
        return dict(results)

async def apply_trusted_reranking(candidates):
    alpha = 0.05 
    citation_map = await fetch_real_citations_batch(candidates)
    reranked = {}
    for pid, score in candidates.items():
        citations = citation_map.get(pid, 0)
        boost = alpha * math.log1p(citations)
        reranked[pid] = score + boost
        state["papers"][pid]["_temp_citations"] = citations
    return reranked

# --- 7.2 GEMINI INTEGRATION (FULL TEXT) ---
def get_full_paper_text(pid):
    try:
        pdf_path = state["papers"][pid]["path"] 
        doc = fitz.open(pdf_path) 
        full_text = ""
        for i, page in enumerate(doc):
            if i > 20: break 
            full_text += page.get_text() + "\n"
        
        full_text = " ".join(full_text.split())
        return full_text[:40000] 
    except Exception as e:
        print(f"Error reading PDF {pid}: {e}")
        return "Full text unavailable."

async def call_gemini_api(papers_list, user_query_raw):
    search_context = user_query_raw if user_query_raw else "Visual Similarity Search"

    context_data = ""
    for i, p in enumerate(papers_list[:3]):
        pid = p['id']
        full_content = get_full_paper_text(pid)
        context_data += f"--- START PAPER {i+1} ---\nTitle: {p['filename']}\nContent: {full_content}\n--- END PAPER {i+1} ---\n\n"

    prompt_text = f"""
    You are an expert Scientific Research Assistant. 
    User Context/Query: "{search_context}"

    Here are the full texts of the top 3 retrieved papers:
    {context_data}

    Task:
    1. For each paper, write a **Detailed Summary** (one long paragraph) covering its core methodology and contribution.
    2. Provide **Key Insights** (bullet points) for each.
    3. Synthesize a **Collective Consensus** paragraph at the end.

    Strict Output Format (HTML Only, NO Markdown):
    <div class="space-y-4">
        <!-- Paper 1 -->
        <div class="bg-gray-800 p-4 rounded-lg border-l-4 border-cyan-500 shadow-lg">
            <h4 class="text-cyan-400 font-bold text-lg mb-2">[Paper 1 Title]</h4>
            <div class="text-gray-300 text-sm mb-3 text-justify">[Long Summary]</div>
            <div class="bg-gray-900/50 p-3 rounded">
                <span class="text-xs font-bold text-gray-400 uppercase tracking-wide">Key Insights</span>
                <ul class="list-disc pl-5 mt-1 text-sm text-gray-400 space-y-1">
                    <li>Insight 1...</li>
                </ul>
            </div>
        </div>
        <!-- Repeat for Paper 2, 3 -->
        <div class="bg-gray-800 p-4 rounded-lg border-l-4 border-cyan-500 shadow-lg">
            <h4 class="text-cyan-400 font-bold text-lg mb-2">[Paper 2 Title]</h4>
            <div class="text-gray-300 text-sm mb-3 text-justify">[Long Summary]</div>
            <div class="bg-gray-900/50 p-3 rounded">
                <span class="text-xs font-bold text-gray-400 uppercase tracking-wide">Key Insights</span>
                <ul class="list-disc pl-5 mt-1 text-sm text-gray-400 space-y-1">
                    <li>Insight 1...</li>
                </ul>
            </div>
        </div>
        <div class="bg-gray-800 p-4 rounded-lg border-l-4 border-cyan-500 shadow-lg">
            <h4 class="text-cyan-400 font-bold text-lg mb-2">[Paper 3 Title]</h4>
            <div class="text-gray-300 text-sm mb-3 text-justify">[Long Summary]</div>
            <div class="bg-gray-900/50 p-3 rounded">
                <span class="text-xs font-bold text-gray-400 uppercase tracking-wide">Key Insights</span>
                <ul class="list-disc pl-5 mt-1 text-sm text-gray-400 space-y-1">
                    <li>Insight 1...</li>
                </ul>
            </div>
        </div>

        <div class="mt-6 p-5 bg-gradient-to-r from-gray-800 to-gray-900 rounded-xl border border-purple-500/30">
            <h4 class="text-purple-400 font-bold text-base mb-2 flex items-center gap-2">
                <span>✨</span> Synthesized Consensus
            </h4>
            <p class="text-sm text-gray-300 leading-relaxed text-justify">
                [Consensus Paragraph]
            </p>
        </div>
    </div>
    """

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            if response.status_code != 200: return f"AI Error: {response.text}"
            data = response.json()
            raw_text = data['candidates'][0]['content']['parts'][0]['text']
            clean_text = re.sub(r"```(html)?", "", raw_text, flags=re.IGNORECASE).strip()
            return clean_text

    except Exception as e:
        return f"⚠️ AI service unavailable: {e}"

# --- FORMATTING ---
def format_results(candidate_list, papers_map, limit, show_citations=False):
    output = []
    if isinstance(candidate_list, list): items = candidate_list[:limit]
    elif isinstance(candidate_list, dict): items = sorted(candidate_list.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    for pid, score in items:
        if pid not in papers_map: continue
        paper_data = papers_map[pid]
        filename = paper_data["filename"]
        citations = paper_data.get("_temp_citations", 0) if show_citations else None
        
        output.append({
            "id": pid, "score": round(score, 3),
            "filename": filename, "citations": citations,
            "url": f"/pdfs/{filename}"
        })
    return output

@app.post("/search")
async def search(
    query: str = Form(None), file: UploadFile = File(None), 
    k: int = 10, trusted_mode: bool = Form(False), ai_summary: bool = Form(False)
):
    if not query and not file: raise HTTPException(400, "Input required")

    device = state["device"]
    text_candidates = {} 
    img_candidates = {}

    if query:
        inputs = state["processor"](text=[query], return_tensors="pt", truncation=True, max_length=8192).to(device)
        with torch.no_grad():
            q_emb = state["model"].get_text_features(**inputs).cpu().numpy()
            q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        D, I = state["index_txt"].search(q_emb, k * 2)
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            pid = state["meta_txt"][idx]["paper_id"]
            if pid in state["papers"]:
                text_candidates[pid] = max(text_candidates.get(pid, 0), float(score))

    if file:
        img_data = await file.read()
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        inputs = state["processor"](images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            q_emb = state["model"].get_image_features(**inputs).cpu().numpy()
            q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        D, I = state["index_img"].search(q_emb, k * 2)
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            pid = state["meta_img"][idx]["paper_id"]
            if pid in state["papers"]:
                img_candidates[pid] = max(img_candidates.get(pid, 0), float(score))

    if trusted_mode:
        if text_candidates: text_candidates = await apply_trusted_reranking(text_candidates)
        if img_candidates: img_candidates = await apply_trusted_reranking(img_candidates)

    response = {"mode": "single", "summary": None}
    final_top_papers = []

    if query and file:
        sorted_text = sorted(text_candidates.items(), key=lambda x: x[1], reverse=True)
        sorted_img = sorted(img_candidates.items(), key=lambda x: x[1], reverse=True)
        combined_list = []
        seen_pids = set()
        max_len = max(len(sorted_text), len(sorted_img))
        for i in range(max_len):
            if i < len(sorted_text):
                pid, score = sorted_text[i]
                if pid not in seen_pids:
                    combined_list.append((pid, score))
                    seen_pids.add(pid)
            if i < len(sorted_img):
                pid, score = sorted_img[i]
                if pid not in seen_pids:
                    combined_list.append((pid, score))
                    seen_pids.add(pid)
        
        response["mode"] = "hybrid"
        response["results_text"] = format_results(text_candidates, state["papers"], k, trusted_mode)
        response["results_image"] = format_results(img_candidates, state["papers"], k, trusted_mode)
        response["results_combined"] = format_results(combined_list, state["papers"], k, trusted_mode)
        final_top_papers = response["results_combined"]
    else:
        candidates = text_candidates if query else img_candidates
        response["results"] = format_results(candidates, state["papers"], k, trusted_mode)
        final_top_papers = response["results"]

    if ai_summary and final_top_papers:
        ai_text = await call_gemini_api(final_top_papers, query)
        response["summary"] = ai_text

    return response