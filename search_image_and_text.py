import sys
import numpy as np
import faiss
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

META_PATH = "dataset/image_filenames.npy"
INDEX_PATH = "dataset/faiss_index.bin"

device = "cuda" if torch.cuda.is_available() else "cpu"
index = faiss.read_index(INDEX_PATH)
filenames = np.load(META_PATH)
model = SentenceTransformer("clip-ViT-B-32", device=device)


def search_by_text(query, k=5):
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(emb, k)
    return [(filenames[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]


def search_by_image(image_path, k=5):
    img = Image.open(image_path).convert("RGB")
    emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(emb[None, :], k)
    return [(filenames[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "guitar"
    if query.lower().endswith((".jpg", ".png")):
        results = search_by_image(query)
    else:
        results = search_by_text(query)
    for fname, score in results:
        print(f"{score:.3f}  {fname}")
