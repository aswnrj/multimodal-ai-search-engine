import os
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "dataset/images"
NUM_IMAGES = 500

os.makedirs(OUT_DIR, exist_ok=True)

dataset = load_dataset("flwrlabs/caltech101", split="train")
dataset = dataset.shuffle(seed=42).select(range(min(NUM_IMAGES, len(dataset))))

for i, item in enumerate(tqdm(dataset, desc="Saving images")):
    label = dataset.features["label"].int2str(item["label"])
    item["image"].save(os.path.join(OUT_DIR, f"{i:05d}_{label}.jpg"))

print(f"Saved {len(dataset)} images to {OUT_DIR}")
