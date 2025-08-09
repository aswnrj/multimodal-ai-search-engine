import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image

# Paths
EMB_PATH = "dataset/image_embeddings.npy"
META_PATH = "dataset/image_filenames.npy"
INDEX_PATH = "dataset/faiss_index.bin"

# Load data
embeddings = np.load(EMB_PATH)
image_files = np.load(META_PATH)

# Create FAISS index (cosine similarity → use inner product with normalized vectors)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"📦 FAISS index built with {index.ntotal} vectors.")

# Save index
faiss.write_index(index, INDEX_PATH)
print(f"✅ Index saved to {INDEX_PATH}")

# --- Test retrieval with text ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("clip-ViT-B-32", device=device)

def search_by_text(query, k=5):
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(query_emb, k)
    return [(image_files[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]

# Example
results = search_by_text("a dog playing")
print("\n🔍 Search results for: 'a dog playing'")
for fname, score in results:
    print(f"{fname} (score={score:.3f})")
