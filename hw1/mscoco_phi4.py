import torch
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# Import NLP Metrics
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

nltk.download('punkt')
nltk.download('wordnet')

# Load dataset (MS COCO 5k test set)
ds = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Phi-4 Multimodal model & processor
model_path = "microsoft/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to(device)

generation_config = GenerationConfig.from_pretrained(model_path)

# Define evaluation functions
def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    scores = [scorer.score(ref, hypothesis) for ref in reference]
    avg_scores = {
        'rouge1': np.mean([s['rouge1'].fmeasure for s in scores]),
        'rouge2': np.mean([s['rouge2'].fmeasure for s in scores])
    }
    return avg_scores

def compute_meteor(reference, hypothesis):
    hypothesis_tokens = hypothesis.split()
    return np.mean([meteor_score([ref.split()], hypothesis_tokens) for ref in reference])

# Lists for corpus-level BLEU
all_references = []
all_hypotheses = []
results = []

# Process images and generate captions
for i, d in enumerate(tqdm(ds['test'], desc="Processing Images")):
    image = d['image']
    reference_captions = d['caption']

    # Define image captioning prompt (Phi-4 requires <|image_1|> token)
    prompt = "<|image_1|>Describe this image in detail."

    # Process image & prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs, max_new_tokens=50, generation_config=generation_config
        )

    generated_caption = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    # Store for corpus BLEU
    tokenized_refs = [ref.split() for ref in reference_captions]
    tokenized_hyp = generated_caption.split()
    all_references.append(tokenized_refs)
    all_hypotheses.append(tokenized_hyp)

    # Compute ROUGE & METEOR
    rouge_scores = compute_rouge(reference_captions, generated_caption)
    meteor_value = compute_meteor(reference_captions, generated_caption)

    results.append({
        "image_id": i,
        "generated_caption": generated_caption,
        "reference_captions": reference_captions,
        "rouge1": rouge_scores['rouge1'],
        "rouge2": rouge_scores['rouge2'],
        "meteor": meteor_value
    })

# Compute Corpus BLEU
smoothie = SmoothingFunction().method1
corpus_bleu_score = corpus_bleu(all_references, all_hypotheses, smoothing_function=smoothie)

# Compute Average ROUGE & METEOR
avg_rouge1 = np.mean([r["rouge1"] for r in results])
avg_rouge2 = np.mean([r["rouge2"] for r in results])
avg_meteor = np.mean([r["meteor"] for r in results])

print(f"Corpus BLEU: {corpus_bleu_score:.4f}")
print(f"Average ROUGE-1: {avg_rouge1:.4f}")
print(f"Average ROUGE-2: {avg_rouge2:.4f}")
print(f"Average METEOR: {avg_meteor:.4f}")

# Save results to JSON
final_output = {
    "corpus_bleu": corpus_bleu_score,
    "average_rouge1": float(avg_rouge1),
    "average_rouge2": float(avg_rouge2),
    "average_meteor": float(avg_meteor),
    "results": results
}

with open('results/mscoco_phi4_result.json', 'w') as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print("✅ Results saved to 'results/mscoco_phi4_result.json'")

