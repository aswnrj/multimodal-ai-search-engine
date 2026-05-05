import os
import time
import numpy as np
import faiss
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer

META_PATH = "dataset/image_filenames.npy"
INDEX_PATH = "dataset/faiss_index.bin"
IMG_DIR = "dataset/images"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}")
model = SentenceTransformer("clip-ViT-B-32", device=device)
index = faiss.read_index(INDEX_PATH)
filenames = np.load(META_PATH)
print(f"Index ready with {index.ntotal} images")


def label_from_filename(fname):
    # filenames look like 00042_guitar.jpg
    base = os.path.basename(fname).rsplit(".", 1)[0]
    return base.split("_", 1)[1] if "_" in base else base


def format_results(scores, idxs, elapsed):
    paths = []
    lines = [f"Search took {elapsed * 1000:.0f} ms"]
    for rank, (i, score) in enumerate(zip(idxs, scores), 1):
        fname = filenames[i]
        paths.append(os.path.join(IMG_DIR, fname))
        lines.append(f"{rank}. {label_from_filename(fname)} ({score:.3f})")
    return paths, "\n".join(lines)


def text_search(query, k):
    if not query.strip():
        return [], "Enter a query."
    t = time.time()
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(emb, k)
    return format_results(scores[0], idxs[0], time.time() - t)


def image_search(image, k):
    if image is None:
        return [], "Upload an image."
    t = time.time()
    emb = model.encode(image.convert("RGB"), convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(emb[None, :], k)
    return format_results(scores[0], idxs[0], time.time() - t)


with gr.Blocks(title="Multimodal Image Search") as demo:
    gr.Markdown("# Multimodal Image Search\nSearch 500 Caltech101 images by text or by image.")

    with gr.Tab("Text"):
        query = gr.Textbox(label="Query", placeholder="a red car")
        k_text = gr.Slider(1, 20, value=5, step=1, label="Results")
        btn_text = gr.Button("Search", variant="primary")
        gallery_text = gr.Gallery(columns=5, height="auto")
        info_text = gr.Markdown()
        gr.Examples(
            ["a cat", "airplane in the sky", "person playing guitar", "red car"],
            inputs=query,
        )

        btn_text.click(text_search, [query, k_text], [gallery_text, info_text])
        query.submit(text_search, [query, k_text], [gallery_text, info_text])

    with gr.Tab("Image"):
        image_in = gr.Image(label="Query image", type="pil")
        k_image = gr.Slider(1, 20, value=5, step=1, label="Results")
        btn_image = gr.Button("Search", variant="primary")
        gallery_image = gr.Gallery(columns=5, height="auto")
        info_image = gr.Markdown()

        btn_image.click(image_search, [image_in, k_image], [gallery_image, info_image])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
