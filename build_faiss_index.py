import faiss
import numpy as np

EMB_PATH = "dataset/image_embeddings.npy"
INDEX_PATH = "dataset/faiss_index.bin"

embeddings = np.load(EMB_PATH)

# Inner product on L2-normalized vectors == cosine similarity
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
print(f"Built index with {index.ntotal} vectors -> {INDEX_PATH}")
