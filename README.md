# Multimodal AI Search Engine

A small image search app that takes either a text query ("a red car") or an image, and returns the most similar images from a local dataset. Built on CLIP for the embeddings and FAISS for the lookup, with a Gradio UI on top.

I built this to learn how CLIP-style models bridge text and images in a shared vector space, and to get hands-on with FAISS. It's a learning project, not a production system — the dataset is intentionally tiny (500 images) so the whole pipeline runs in a few minutes on a laptop.

## What it does

- Type a description, get back the closest matching images.
- Upload an image, get back visually similar ones from the dataset.
- Both modes go through the same 512-dim CLIP embedding space, so they share one FAISS index.

The dataset is 500 images sampled from Caltech101 (101 object categories — guitars, cars, dogs, airplanes, etc.).

## Setup

You'll need Python 3.8+ and a few GB of disk for the model cache. A GPU helps but isn't required — CPU inference for 500 images takes maybe a minute.

```bash
git clone https://github.com/aswnrj/multimodal-ai-search-engine.git
cd multimodal-ai-search-engine
pip install -r requirements.txt
```

Then run the three setup scripts in order. They download the dataset, embed the images with CLIP, and build the FAISS index:

```bash
python download_images_hf.py     # ~500 images into dataset/images/
python embed_images_clip.py      # writes image_embeddings.npy + image_filenames.npy
python build_faiss_index.py      # writes faiss_index.bin
```

Once that's done, launch the app:

```bash
python app.py
```

It'll be at http://localhost:7860.

## How it works

CLIP is trained so that an image and a matching caption end up near each other in the same vector space. That means I can encode the dataset images once, encode the query (text or image) at search time, and use cosine similarity to rank matches. FAISS handles the similarity search.

A couple of implementation notes:

- All embeddings are L2-normalized at encode time, so cosine similarity reduces to a plain inner product. That's why the index is `IndexFlatIP` and not some custom cosine variant.
- `IndexFlatIP` is exhaustive (brute force). With only 500 vectors that's perfectly fine — at this scale, anything fancier (HNSW, IVF) is wasted complexity. The bottleneck is CLIP encoding the query, not the FAISS lookup.
- The image filenames are saved as `{idx:05d}_{class_label}.jpg`. `app.py` parses the class label out of the filename when displaying results, so don't rename the files.

## File layout

| File | What it does |
|------|--------------|
| `download_images_hf.py` | Pulls Caltech101 from HuggingFace and saves 500 shuffled images locally. |
| `embed_images_clip.py` | Runs every image through CLIP-ViT-B-32 and saves the embeddings + filenames as `.npy` files. |
| `build_faiss_index.py` | Loads the embeddings, builds an inner-product FAISS index, writes it to disk. |
| `search_image_and_text.py` | Standalone CLI that does text and image search without the UI. Handy for debugging. |
| `app.py` | The Gradio web UI. Loads the index and model once at startup. |
| `requirements.txt` | Dependencies. |

The `dataset/` folder is gitignored — you regenerate it locally with the three scripts above.

## Stack

- **CLIP-ViT-B-32** via `sentence-transformers` for embeddings.
- **FAISS** (CPU build by default; swap to `faiss-gpu` if you want, but at this dataset size it doesn't matter).
- **Gradio** for the UI.
- **HuggingFace `datasets`** for pulling Caltech101.
