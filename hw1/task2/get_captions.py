import os
import json
import numpy as np
import requests
import torch
from glob import glob
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from datasets import load_dataset
from tqdm import tqdm

VERBOSE = True

model_path = "microsoft/Phi-4-multimodal-instruct"

kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto',
    _attn_implementation='flash_attention_2',
).cuda()

# Load all image file paths
image_folder = 'content_image'
image_paths = sorted(glob(os.path.join(image_folder, '*.jpg')))

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

results = []
prompt = f'{user_prompt}<|image_1|>Carefully caption the avatar in the image. Do not describe the background.{prompt_suffix}{assistant_prompt}'
# prompt = f'{user_prompt}<|image_1|>Carefully caption the avatar in the image as more detail as possible.{prompt_suffix}{assistant_prompt}'
# prompt = f'{user_prompt}<|image_1|>Caption the avatar in the image in one sentence.{prompt_suffix}{assistant_prompt}'
results.append({"prompt": prompt})

generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

batch_size = 10

for i in tqdm(range(0, len(image_paths), batch_size), desc="Captioning Images in Batches"):
    batch_paths = image_paths[i:i + batch_size]
    batch_images = [Image.open(p).convert('RGB') for p in batch_paths]
    batch_names = [os.path.basename(p) for p in batch_paths]

    # Same prompt for each image in batch
    batch_prompts = [prompt] * len(batch_images)

    inputs = processor(text=batch_prompts, images=batch_images, return_tensors='pt', padding=True).to('cuda:0')
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=1000,
        generation_config=generation_config,
    )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    generated_captions = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    for name, caption in zip(batch_names, generated_captions):
        results.append({
            "image_name": name,
            "generated_caption": caption
        })
        if VERBOSE:
            print('>' * 50)
            print(f"Processing Image: {name}")
            print('-' * 50)
            print(caption)
            print('<' * 50)

# Save results to JSON
os.makedirs("results", exist_ok=True)
result_fn = 'results/avatar_only.json'
with open(result_fn, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Captions saved to '{result_fn}'")

