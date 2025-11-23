import os
import json
import faiss
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import AutoModel, AutoProcessor
from PIL import Image
import io

# --- PATH CONFIGURATION (Relative to project root) ---
EMB_DIR = "outputs/embeddings"
IDX_DIR = "outputs/indexes"
PAPER_MAP_FILE = "outputs/papers.json"
PDF_REPO = "pdf_repo" 
MODEL_ID = "jinaai/jina-clip-v1"

app = FastAPI(title="Research Engine")

# 1. Allow Frontend to talk to Backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Serve PDFs so the browser can download/view them
# URL like: http://localhost:8000/pdfs/my_paper.pdf
app.mount("/pdfs", StaticFiles(directory=PDF_REPO), name="pdfs")

# --- GLOBAL STATE ---
state = {
    "model": None,
    "processor": None,
    "index_txt": None,
    "index_img": None,
    "meta_txt": [],
    "meta_img": [],
    "papers": {}
}

@app.on_event("startup")
async def load_resources():
    print("--- STARTING SERVER ---")
    print(f"Loading Jina-CLIP ({MODEL_ID})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["device"] = device
    
    # Load Model
    state["model"] = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    state["processor"] = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading FAISS Indexes...")
    state["index_txt"] = faiss.read_index(f"{IDX_DIR}/faiss_text.index")   # <--- CORRECT NAME
    state["index_img"] = faiss.read_index(f"{IDX_DIR}/faiss_image.index")  # <--- CORRECT NAME

    print("Loading Metadata & Paper Map...")
    with open(f"{EMB_DIR}/text_metadata.json", "r") as f:
        state["meta_txt"] = json.load(f)
    with open(f"{EMB_DIR}/image_metadata.json", "r") as f:
        state["meta_img"] = json.load(f)
    with open(PAPER_MAP_FILE, "r") as f:
        state["papers"] = json.load(f)
    
    print("✅ Server Ready! Go to http://localhost:8000")

@app.get("/")
def serve_ui():
    # Serves the HTML frontend
    return FileResponse("appui/index.html")
@app.post("/search")
async def search(
    query: str = Form(None), 
    file: UploadFile = File(None),
    k: int = 10
):
    if not query and not file:
        raise HTTPException(400, "Enter text or upload an image.")

    device = state["device"]
    
    # Dictionary to hold scores: {pid: {'text': 0.0, 'img': 0.0}}
    scores_map = {} 

    # We fetch more candidates (k*10) to ensure the correct paper isn't filtered out early
    # simply because its score in one modality was low.
    search_k = k * 10 

    # --- 1. TEXT SEARCH ---
    if query:
        inputs = state["processor"](text=[query], return_tensors="pt", truncation=True, max_length=8192).to(device)
        with torch.no_grad():
            q_emb = state["model"].get_text_features(**inputs).cpu().numpy()
            q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        
        D, I = state["index_txt"].search(q_emb, search_k)
        
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            meta = state["meta_txt"][idx]
            pid = meta["paper_id"]
            if pid in state["papers"]:
                if pid not in scores_map: scores_map[pid] = {'text': 0.0, 'img': 0.0}
                scores_map[pid]['text'] = max(scores_map[pid]['text'], float(score))

    # --- 2. IMAGE SEARCH ---
    if file:
        img_data = await file.read()
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        inputs = state["processor"](images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            q_emb = state["model"].get_image_features(**inputs).cpu().numpy()
            q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        D, I = state["index_img"].search(q_emb, search_k)

        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            meta = state["meta_img"][idx]
            pid = meta["paper_id"]
            if pid in state["papers"]:
                if pid not in scores_map: scores_map[pid] = {'text': 0.0, 'img': 0.0}
                scores_map[pid]['img'] = max(scores_map[pid]['img'], float(score))

    # --- 3. COMBINE SCORES (Soft OR Logic) ---
    final_scores = []

    for pid, vals in scores_map.items():
        t_score = vals['text']
        i_score = vals['img']
        
        # LOGIC: Take the best single match, but give a small bonus if it matches both.
        # This allows Paper A (High Text) and Paper B (High Image) to both rank high.
        # But Paper C (Medium Text + Medium Image) gets a small boost.
        
        base_score = max(t_score, i_score)
        bonus = 0.1 * min(t_score, i_score) # Bonus only if both exist
        
        combined_score = base_score + bonus
        
        final_scores.append((pid, combined_score))

    # Sort
    ranked_results = sorted(final_scores, key=lambda x: x[1], reverse=True)[:k]

    response_data = []
    for pid, score in ranked_results:
        paper_data = state["papers"][pid]
        filename = paper_data["filename"]
        pdf_url = f"/pdfs/{filename}"
        
        response_data.append({
            "id": pid,
            "score": round(score, 3),
            "filename": filename,
            "url": pdf_url
        })

    return {"results": response_data}