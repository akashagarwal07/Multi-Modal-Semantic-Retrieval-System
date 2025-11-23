import fitz  # PyMuPDF
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch
import gc
import glob

MODEL_PATH = "yolov8l-doclaynet.pt" 
_model_instance = None

def load_yolo_model():
    global _model_instance
    if _model_instance is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
        print("Loading YOLO model to GPU...")
        _model_instance = YOLO(MODEL_PATH)
    return _model_instance

def unload_yolo_model():
    global _model_instance
    if _model_instance is not None:
        del _model_instance
        _model_instance = None
        gc.collect()
        torch.cuda.empty_cache()
        print("YOLO model unloaded.")

def extract_from_pdf(pdf_path, output_text_dir, output_img_dir, paper_id):
    text_filename = f"{paper_id}.txt"
    text_save_path = os.path.join(output_text_dir, text_filename)

    # --- SCENARIO A: ALREADY PROCESSED ---
    if os.path.exists(text_save_path):
        if os.path.getsize(text_save_path) > 0:
            existing_images = glob.glob(os.path.join(output_img_dir, f"{paper_id}_*.png"))
            return text_save_path, existing_images

    # --- SCENARIO B: NEW PAPER ---
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return None, []

    # 1. Text Extraction (Robust Mode)
    full_text = ""
    for i, page in enumerate(doc):
        try:
            # --- FIX FOR CRASH ---
            full_text += page.get_text() + "\n"
        except Exception as e:
            # Log error but continue to next page
            print(f"Warning: Skipping text on page {i} of {paper_id} (corruption error).")
            continue
    
    with open(text_save_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    # 2. Prepare Images (Zoom 1.33)
    zoom = 1.5
    mat = fitz.Matrix(zoom, zoom)
    
    page_images = []
    page_indices = []

    for i, page in enumerate(doc):
        try:
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 4: 
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1: 
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                
            page_images.append(img_array)
            page_indices.append(i)
        except Exception:
            # If image render fails, skip page
            continue 
    doc.close()

    new_image_paths = []

    # 3. GPU Batch Inference (Chunked)
    if page_images:
        model = load_yolo_model()
        
        chunk_size = 10
        total_pages = len(page_images)

        for start_idx in range(0, total_pages, chunk_size):
            end_idx = min(start_idx + chunk_size, total_pages)
            
            batch_imgs = page_images[start_idx:end_idx]
            batch_indices = page_indices[start_idx:end_idx]
            
            try:
                results = model.predict(batch_imgs, conf=0.25, imgsz=1024, batch=8, verbose=False, device=0)

                for i, result in enumerate(results):
                    page_idx = batch_indices[i]
                    original_img = batch_imgs[i]
                    
                    local_save_count = 0
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        class_name = result.names[cls_id].lower()

                        if "picture" in class_name or "table" in class_name or "figure" in class_name:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            if (x2 - x1) < 50 or (y2 - y1) < 50: continue
                            
                            h, w, _ = original_img.shape
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            crop = original_img[y1:y2, x1:x2]
                            
                            try:
                                crop_img = Image.fromarray(crop)
                                img_filename = f"{paper_id}_p{page_idx}_{class_name}_{local_save_count}.png"
                                save_path = os.path.join(output_img_dir, img_filename)
                                crop_img.save(save_path)
                                
                                new_image_paths.append(save_path)
                                local_save_count += 1
                            except Exception:
                                continue
            
            except torch.cuda.OutOfMemoryError:
                print(f"OOM on chunk {start_idx}-{end_idx} of {paper_id}. Skipping chunk.")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"Error processing images for {paper_id}: {e}")
                continue

    del page_images
    gc.collect()
    
    return text_save_path, new_image_paths