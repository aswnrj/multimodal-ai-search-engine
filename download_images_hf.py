import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

OUT_DIR = "dataset/images"
NUM_IMAGES = 500

os.makedirs(OUT_DIR, exist_ok=True)

print("📥 Loading Caltech101 dataset...")
dataset = load_dataset("flwrlabs/caltech101", split="train")

# Shuffle and select subset
dataset = dataset.shuffle(seed=42).select(range(min(NUM_IMAGES, len(dataset))))

print(f"💾 Saving {len(dataset)} images locally...")
for i, item in enumerate(tqdm(dataset)):
    img = item["image"]
    label = item["label"]
    label_name = dataset.features["label"].int2str(label)
    fname = f"{i:05d}_{label_name}.jpg"
    img.save(os.path.join(OUT_DIR, fname))

print(f"✅ Done! Saved {len(dataset)} images to {OUT_DIR}")
