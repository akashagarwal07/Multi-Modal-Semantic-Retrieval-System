import faiss
import numpy as np

class SearchEngine:
    def __init__(self, index_text, index_image, txt_meta, img_meta):
        self.index_text = faiss.read_index(index_text)
        self.index_image = faiss.read_index(index_image)

        self.text_meta = txt_meta
        self.image_meta = img_meta

    def search_fusion(self, q_txt_emb, q_img_emb, topk=10, alpha=0.5):
        # Search separately
        D_txt, I_txt = self.index_text.search(q_txt_emb, topk)
        D_img, I_img = self.index_image.search(q_img_emb, topk)

        scores = {}

        # Combine text results
        for score, idx in zip(D_txt[0], I_txt[0]):
            file = self.text_meta[idx]["file"]
            scores[file] = scores.get(file, 0) + alpha * score

        # Combine image results
        for score, idx in zip(D_img[0], I_img[0]):
            file = self.image_meta[idx]["file"]
            scores[file] = scores.get(file, 0) + (1 - alpha) * score

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:topk]
