import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image

# Paths
META_PATH = "dataset/image_filenames.npy"
INDEX_PATH = "dataset/faiss_index.bin"
IMG_DIR = "dataset/images"

# Load index & metadata
index = faiss.read_index(INDEX_PATH)
image_files = np.load(META_PATH)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("clip-ViT-B-32", device=device)

def search_by_text(query, k=5):
    """Search for images matching a text query."""
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(query_emb, k)
    return [(image_files[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]

def search_by_image(image_path, k=5):
    """Search for images visually similar to the given image."""
    img = Image.open(image_path).convert("RGB")
    query_emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
    query_emb = np.expand_dims(query_emb, axis=0)  # FAISS expects [batch, dim]
    scores, idxs = index.search(query_emb, k)
    return [(image_files[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]

# --- Test both ---
print("\n🔍 Text Search: 'guitar'")
for fname, score in search_by_text("guitar"):
    print(f"{fname} (score={score:.3f})")

print("\n🔍 Image Search: using first dataset image")
test_img = f"{IMG_DIR}/{image_files[0]}"
for fname, score in search_by_image(test_img):
    print(f"{fname} (score={score:.3f})")
