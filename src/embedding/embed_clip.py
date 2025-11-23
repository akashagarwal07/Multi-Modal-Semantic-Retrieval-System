import torch
from PIL import Image
import numpy as np
import json
import os
import re
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# Using Jina CLIP (State of the Art, 8k token limit)
MODEL_ID = "jinaai/jina-clip-v1"

def get_text_chunk(text, limit_chars=6000):
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = " ".join(text.split())
    return text[:limit_chars]

def extract_id_from_filename(filename):
    """
    Extracts 'paper_0' from 'paper_0.txt' or 'paper_0_p1_figure.png'
    """
    base = os.path.basename(filename)
    if ".txt" in base:
        return base.replace(".txt", "")
    else:
        # For images like: paper_0_p1_figure...
        # We split by "_p" (page indicator) to get the ID part
        return base.split("_p")[0]

def embed_all(text_file_paths, image_file_paths, save_dir):
    print(f"Loading Jina-CLIP model ({MODEL_ID})...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    print(f"Model loaded on {device}.")

    text_embeds = []
    image_embeds = []
    text_meta = []
    image_meta = []

    # --- 1. EMBED TEXT ---
    print(f"Embedding {len(text_file_paths)} text files...")
    batch_size = 12
    for i in tqdm(range(0, len(text_file_paths), batch_size)):
        batch_paths = text_file_paths[i : i + batch_size]
        batch_texts = []
        valid_batch_paths = []

        for p in batch_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                        clean_content = get_text_chunk(raw_text, limit_chars=6000)
                        batch_texts.append(clean_content)
                        valid_batch_paths.append(p)
                except Exception:
                    pass

        if not batch_texts: continue

        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        batch_np = embeddings.cpu().numpy()
        for k, text_content in enumerate(batch_texts):
            text_embeds.append(batch_np[k:k+1])
            
            # --- KEY CHANGE: ADD PAPER ID TO METADATA ---
            p_id = extract_id_from_filename(valid_batch_paths[k])
            text_meta.append({
                "paper_id": p_id,      # LINK TO MAIN PDF
                "file": valid_batch_paths[k], 
                "preview": text_content[:200] 
            })

    # --- 2. EMBED IMAGES ---
    print(f"Embedding {len(image_file_paths)} images...")
    img_batch_size = 32
    for i in tqdm(range(0, len(image_file_paths), img_batch_size)):
        batch_paths = image_file_paths[i : i + img_batch_size]
        batch_images = []
        valid_batch_paths = []

        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(image)
                valid_batch_paths.append(img_path)
            except Exception:
                pass

        if not batch_images: continue

        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        batch_np = embeddings.cpu().numpy()
        for k in range(len(valid_batch_paths)):
            image_embeds.append(batch_np[k:k+1])
            
            # --- KEY CHANGE: ADD PAPER ID TO METADATA ---
            p_id = extract_id_from_filename(valid_batch_paths[k])
            image_meta.append({
                "paper_id": p_id,        # LINK TO MAIN PDF
                "file": valid_batch_paths[k]
            })

    # --- 3. SAVE ---
    print("Saving embeddings and enriched metadata...")
    if text_embeds:
        np.save(f"{save_dir}/text_embeds.npy", np.vstack(text_embeds))
        with open(f"{save_dir}/text_metadata.json", "w") as f:
            json.dump(text_meta, f)
    
    if image_embeds:
        np.save(f"{save_dir}/image_embeds.npy", np.vstack(image_embeds))
        with open(f"{save_dir}/image_metadata.json", "w") as f:
            json.dump(image_meta, f)