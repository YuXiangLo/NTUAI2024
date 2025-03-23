import json
import os
from PIL import Image
from tqdm import tqdm
import torch
from diffusers import DiffusionPipeline

# === Load the pipeline ===
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# === Paths ===
input_json = "results/avatar_only.json"
output_dir, _ = os.path.splitext(input_json)
os.makedirs(output_dir, exist_ok=True)

# === Load captions ===
with open(input_json, "r") as f:
    data = json.load(f)[1:]

# === Batch parameters ===
batch_size = 4  # You can increase depending on your GPU VRAM

# === Generate images in batches ===
for i in tqdm(range(0, len(data), batch_size), desc="Generating images in batches"):
    batch = data[i:i + batch_size]
    captions = [item["generated_caption"] for item in batch]
    image_names = [item["image_name"] for item in batch]

    # Generate all images in the batch
    images = pipe(captions).images  # Batch inference

    # Save each image
    for img, name in zip(images, image_names):
        output_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_diff3.jpg")
        img.save(output_path)

