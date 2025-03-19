from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import numpy as np
import nltk

# Import corpus_bleu from NLTK
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm

VERBOSE = False

# Load dataset
ds = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)

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
    # Tokenize hypothesis to fix TypeError
    hypothesis_tokens = hypothesis.split()
    return np.mean([meteor_score([ref.split()], hypothesis_tokens) for ref in reference])

# Prepare lists for corpus-level metrics
all_references = []  # Each element: list of reference-token-lists (one image’s references)
all_hypotheses = []  # Each element: tokenized hypothesis for that image

# Also track per-sample scores if you still want them
results = []

for i, d in enumerate(tqdm(ds['test'], desc="Processing Images")):
    image = d['image']
    reference_captions = d['caption']

    # Preprocess and generate caption
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs)
    generated_caption = processor.batch_decode(output, skip_special_tokens=True)[0]

    # Save the tokenized references/hypothesis for corpus BLEU
    tokenized_refs = [ref.split() for ref in reference_captions]
    tokenized_hyp = generated_caption.split()
    all_references.append(tokenized_refs)
    all_hypotheses.append(tokenized_hyp)

    # If you want per-sample scores for ROUGE/METEOR (averaging them later is fine):
    rouge_scores = compute_rouge(reference_captions, generated_caption)
    meteor_score_val = compute_meteor(reference_captions, generated_caption)

    results.append({
        "image_id": i,
        "generated_caption": generated_caption,
        "reference_captions": reference_captions,
        "rouge1": rouge_scores['rouge1'],
        "rouge2": rouge_scores['rouge2'],
        "meteor": meteor_score_val
    })

    if VERBOSE:
        print('>' * 60)
        print(f'Processing {i + 1}/5000')
        print('-' * 60)
        print(generated_caption)
        print('-' * 60)
        print('\n'.join(reference_captions))
        print('<' * 60)

# Now compute corpus BLEU
smoothie = SmoothingFunction().method1
corpus_bleu_score = corpus_bleu(all_references, all_hypotheses, smoothing_function=smoothie)

# You could also average per-image ROUGE/METEOR if you like:
avg_rouge1 = np.mean([r["rouge1"] for r in results])
avg_rouge2 = np.mean([r["rouge2"] for r in results])
avg_meteor = np.mean([r["meteor"] for r in results])

print(f"Corpus BLEU: {corpus_bleu_score:.4f}")
print(f"Average ROUGE-1: {avg_rouge1:.4f}")
print(f"Average ROUGE-2: {avg_rouge2:.4f}")
print(f"Average METEOR: {avg_meteor:.4f}")

# Save everything to JSON (including corpus BLEU if you like)
import json
final_output = {
    "corpus_bleu": corpus_bleu_score,
    "average_rouge1": float(avg_rouge1),
    "average_rouge2": float(avg_rouge2),
    "average_meteor": float(avg_meteor),
    "results": results
}

with open('results/mscoco_result.json', 'w') as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

