import faiss
import numpy as np
import json
import os

def build_faiss_indexes(embedding_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    text_emb = np.load(f"{embedding_dir}/text_embeds.npy").astype("float32")
    img_emb = np.load(f"{embedding_dir}/image_embeds.npy").astype("float32")

    dim = text_emb.shape[1]

    # Build indexes
    index_text = faiss.IndexFlatIP(dim)
    index_image = faiss.IndexFlatIP(dim)

    index_text.add(text_emb)
    index_image.add(img_emb)

    faiss.write_index(index_text, f"{output_dir}/faiss_text.index")
    faiss.write_index(index_image, f"{output_dir}/faiss_image.index")
