import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

# For progress bar
from tqdm import tqdm

# Import NLP metrics
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

# Uncomment if you're running this for the first time and need to download NLTK data
# nltk.download('punkt')
# nltk.download('wordnet')

# Define metric functions
def compute_rouge(reference_list, hypothesis):
    """
    reference_list: list of possible ground-truth references (strings)
    hypothesis: generated string
    Returns: A dict with 'rouge1' and 'rouge2' f-measures
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    scores = [scorer.score(ref, hypothesis) for ref in reference_list]
    avg_scores = {
        'rouge1': np.mean([s['rouge1'].fmeasure for s in scores]),
        'rouge2': np.mean([s['rouge2'].fmeasure for s in scores])
    }
    return avg_scores

def compute_meteor(reference_list, hypothesis):
    """
    reference_list: list of possible ground-truth references (strings)
    hypothesis: generated string
    Returns: METEOR score (float)
    """
    hypothesis_tokens = hypothesis.split()
    return np.mean([meteor_score([ref.split()], hypothesis_tokens) for ref in reference_list])


#####################
# MODEL PREPARATION #
#####################

# Replace with your model path
model_path = "microsoft/Phi-4-multimodal-instruct"

# Define user/assistant prompts for context
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

print("Loading processor and model...")

# Load processor
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto',
    _attn_implementation='flash_attention_2',
).cuda()

# Load generation config
generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

print("Model and processor loaded.")

#################
# DATA PREP     #
#################

print("Loading Flickr30k dataset...")
ds = load_dataset("nlphuji/flickr30k")
test_dataset = ds["test"]  # The test split of Flickr30k

def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    Each item in batch is a dict with keys:
        'image': PIL.Image
        'caption': list of ground-truth reference captions
    """
    # Prepare the text prompts for the entire batch
    prompts = [
        f"{user_prompt}<|image_1|>Caption the image in one sentence.{prompt_suffix}{assistant_prompt}"
        for _ in batch
    ]
    
    # Extract images and references
    images = [item["image"] for item in batch]
    references = [item["caption"] for item in batch]
    
    # Process the batch with the model's processor
    # The processor can handle multiple images + text, returning tensors
    encoding = processor(
        text=prompts,
        images=images,
        return_tensors='pt',
        padding=True
    )
    
    return encoding, references

# Create the DataLoader
batch_size = 4  # Adjust based on available GPU memory
dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=False
)

##################
# INFERENCE LOOP #
##################

all_references = []  # For corpus-level BLEU
all_hypotheses = []
results = []

print("Running inference on test dataset...")

idx_offset = 0
for batch_encoding, ref_captions in tqdm(dataloader, desc="Processing Batches"):
    # Move encoding to GPU
    for k, v in list(batch_encoding.items()):
        if v is None:
            del batch_encoding[k]
        else:
            batch_encoding[k] = v.cuda()

    with torch.no_grad():
        generate_ids = model.generate(
            **batch_encoding,
            max_new_tokens=1000,
            generation_config=generation_config
        )
    
    # Slice off the prompt portion
    prompt_length = batch_encoding["input_ids"].shape[1]
    generate_ids = generate_ids[:, prompt_length:]

    # Decode
    captions = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Post-process each item in the batch
    for i, (refs, hyp) in enumerate(zip(ref_captions, captions)):
        # For BLEU
        tokenized_refs = [r.split() for r in refs]  # list of lists
        tokenized_hyp = hyp.split()
        all_references.append(tokenized_refs)
        all_hypotheses.append(tokenized_hyp)

        # Compute ROUGE & METEOR for this sample
        rouge_scores = compute_rouge(refs, hyp)
        meteor_value = compute_meteor(refs, hyp)

        # Store result
        results.append({
            "image_id": idx_offset + i,
            "generated_caption": hyp,
            "reference_captions": refs,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "meteor": meteor_value
        })

    idx_offset += len(ref_captions)

#############################
# AGGREGATE METRICS & SAVE  #
#############################

# Corpus-level BLEU
smoothie = SmoothingFunction().method1
corpus_bleu_score = corpus_bleu(all_references, all_hypotheses, smoothing_function=smoothie)

# Averages for ROUGE and METEOR
avg_rouge1 = np.mean([r["rouge1"] for r in results])
avg_rouge2 = np.mean([r["rouge2"] for r in results])
avg_meteor = np.mean([r["meteor"] for r in results])

print("\n===== Evaluation Results =====")
print(f"Corpus BLEU:      {corpus_bleu_score:.4f}")
print(f"Average ROUGE-1:  {avg_rouge1:.4f}")
print(f"Average ROUGE-2:  {avg_rouge2:.4f}")
print(f"Average METEOR:   {avg_meteor:.4f}")

# Prepare final output dictionary
final_output = {
    "corpus_bleu": corpus_bleu_score,
    "average_rouge1": float(avg_rouge1),
    "average_rouge2": float(avg_rouge2),
    "average_meteor": float(avg_meteor),
    "results": results
}

# Create output dir if needed
os.makedirs('results', exist_ok=True)
output_path = 'results/flickr_phi4_result.json'
with open(output_path, 'w') as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print(f"✅ Results saved to '{output_path}'")

