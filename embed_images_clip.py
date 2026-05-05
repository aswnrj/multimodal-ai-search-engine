import os
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

IMG_DIR = "dataset/images"
EMB_PATH = "dataset/image_embeddings.npy"
META_PATH = "dataset/image_filenames.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("clip-ViT-B-32", device=device)

image_files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png")))
print(f"Encoding {len(image_files)} images")

embeddings = []
for fname in tqdm(image_files):
    img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
    emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
    embeddings.append(emb)

np.save(EMB_PATH, np.array(embeddings, dtype="float32"))
np.save(META_PATH, np.array(image_files))
print(f"Saved embeddings to {EMB_PATH}")
