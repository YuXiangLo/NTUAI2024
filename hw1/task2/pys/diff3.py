import json
import os
from PIL import Image
from tqdm import tqdm
import torch
from diffusers import StableDiffusion3Pipeline

# === Load the pipeline ===
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# === Paths ===
input_json = "results/captions/simple_prompt.json"
output_dir, _ = os.path.splitext(input_json)
# output_dir = "results/test"
os.makedirs(output_dir, exist_ok=True)

# === Load captions ===
with open(input_json, "r") as f:
    data = json.load(f)[1:]

# === Batch parameters ===
batch_size = 10

# snoopy_prompt = "Generate an image in the style of the classic 'Peanuts' comic based on the following caption:\n\n"
snoopy_prompt = "Peanuts comic style illustration of:\n\n"

# === Generate images in batches ===
for i in tqdm(range(0, len(data), batch_size), desc="Generating images in batches"):
    batch = data[i:i + batch_size]
    captions = [snoopy_prompt + item["generated_caption"] for item in batch]
    image_names = [item["image_name"] for item in batch]

    # Generate all images in the batch
    images = pipe(captions).images  # Batch inference

    # Save each image
    for img, name in zip(images, image_names):
        output_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_diff3.jpg")
        img.save(output_path)

