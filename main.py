import os
# FIX MEMORY FRAGMENTATION (Must be before importing torch)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import json
import torch
import gc
from tqdm import tqdm

# Import the cleanup function
from src.extract.pdf_extractor import extract_from_pdf, unload_yolo_model
from src.embedding.embed_clip import embed_all
from src.indexing.build_indexes import build_faiss_indexes

# --- CONFIGURATION ---
PDF_DIR = "pdf_repo"
OUT_TEXT = "outputs/extracted/texts"
OUT_IMG = "outputs/extracted/images"
EMB_DIR = "outputs/embeddings"
IDX_DIR = "outputs/indexes"
PAPER_MAP_FILE = "outputs/papers.json"

# Ensure output directories exist
os.makedirs(OUT_TEXT, exist_ok=True)
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)

def run_pipeline():
    pdfs = glob.glob(f"{PDF_DIR}/*.pdf")
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}")
        return

    text_file_paths = []
    image_file_paths = []
    paper_mapping = {} 

    print(f"=== Extracting PDFs ({len(pdfs)} files) ===")
    
    for i, pdf_path in enumerate(tqdm(pdfs)):
        paper_id = f"paper_{i}"
        
        # 1. Map ID
        paper_mapping[paper_id] = {
            "path": pdf_path,
            "filename": os.path.basename(pdf_path)
        }

        # 2. Run Extractor
        result = extract_from_pdf(pdf_path, OUT_TEXT, OUT_IMG, paper_id)
        
        if result:
            txt_path, img_paths = result
            if txt_path: 
                text_file_paths.append(txt_path)
            if img_paths:
                image_file_paths.extend(img_paths)

    # --- CRITICAL MEMORY CLEANUP ---
    print("Cleaning up Extraction resources...")
    unload_yolo_model() # Remove YOLO from GPU
    gc.collect()        # Clear System RAM
    torch.cuda.empty_cache() # Clear VRAM
    print(f"GPU Memory Free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

    # --- SAVE MAPPING ---
    print(f"Saving paper mapping to {PAPER_MAP_FILE}...")
    with open(PAPER_MAP_FILE, "w") as f:
        json.dump(paper_mapping, f, indent=4)

    print("\n=== Generating Jina-CLIP Embeddings ===")
    # Now we have empty VRAM for the huge Jina model
    embed_all(text_file_paths, image_file_paths, EMB_DIR)

    print("\n=== Building FAISS Indexes ===")
    build_faiss_indexes(EMB_DIR, IDX_DIR)

    print("\nPipeline completed!")

if __name__ == "__main__":
    run_pipeline()