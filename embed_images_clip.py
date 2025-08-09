import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# Paths
IMG_DIR = "dataset/images"
EMB_PATH = "dataset/image_embeddings.npy"
META_PATH = "dataset/image_filenames.npy"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Using device: {device}")
model = SentenceTransformer("clip-ViT-B-32", device=device)

# Load images
image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png"))]
print(f"🖼 Found {len(image_files)} images in {IMG_DIR}")

embeddings = []
for fname in tqdm(image_files, desc="Encoding images"):
    img_path = os.path.join(IMG_DIR, fname)
    img = Image.open(img_path).convert("RGB")
    emb = model.encode(img, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    embeddings.append(emb)

embeddings = np.array(embeddings, dtype="float32")

# Save
np.save(EMB_PATH, embeddings)
np.save(META_PATH, np.array(image_files))
print(f"✅ Saved embeddings to {EMB_PATH}")
print(f"✅ Saved filenames to {META_PATH}")
