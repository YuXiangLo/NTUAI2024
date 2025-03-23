import os
import json
import torch
from glob import glob
from PIL import Image, ImageOps
from diffusers import StableDiffusionImg2ImgPipeline
from tqdm import tqdm

strength = 0.8
batch_size = 10

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Load your prompts
snoopy_prompt = "Peanuts comic style illustration of:\n\n"
with open('results/captions/avatar_only.json', 'r') as f:
    data = json.load(f)[1:]  # Skipping the first item with [1:]
prompts = [snoopy_prompt + item['generated_caption'] for item in data]

# Get your input images
image_folder = 'content_image'
image_paths = sorted(glob(os.path.join(image_folder, '*.jpg')))

# Make sure "results/test" folder exists
output_dir = "results/task2_2"
os.makedirs(output_dir, exist_ok=True)

def prepare_image(image_path, target_size=(512, 512)):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # If either dimension < 512, pad; otherwise, resize down
    if width < target_size[0] or height < target_size[1]:
        img = ImageOps.pad(img, target_size, color=(0, 0, 0))  # black padding
    else:
        img = img.resize(target_size)
    return img

# Process in batches
for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches"):
    batch_paths = image_paths[i : i + batch_size]
    batch_prompts = prompts[i : i + batch_size]

    batch_images = [prepare_image(p) for p in batch_paths]

    # Run inference
    images = pipe(
        prompt=batch_prompts,
        image=batch_images,
        strength=strength
    ).images  # This is already a list of PIL Images

    # Save each result
    for path, img in zip(batch_paths, images):
        # Example: results/test/<original_filename>.png
        filename = os.path.basename(path).rsplit('.', 1)[0]  # remove extension
        out_path = os.path.join(output_dir, filename + "_styled.png")
        img.save(out_path)

