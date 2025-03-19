import json
import evaluate
import string
from tqdm import tqdm

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Load evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

# Load JSON results
with open('results/mscoco_phi4_result.json', 'r') as f:
    data = json.load(f)

# Extract and preprocess references and predictions
all_references = [[preprocess_text(ref) for ref in item['reference_captions']] for item in data['results']]
all_predictions = [preprocess_text(item['generated_caption']) for item in data['results']]

# Recompute metrics
bleu_score = bleu.compute(predictions=all_predictions, references=all_references)
rouge_scores = rouge.compute(predictions=all_predictions, references=[r[0] for r in all_references])
meteor_score = meteor.compute(predictions=all_predictions, references=[r[0] for r in all_references])

# Print updated results
print(f"Corpus BLEU: {bleu_score['bleu']:.4f}")
print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
print(f"METEOR: {meteor_score['meteor']:.4f}")

# Update JSON with new results
data.update({
    "corpus_bleu": bleu_score['bleu'],
    "rouge1": rouge_scores['rouge1'],
    "rouge2": rouge_scores['rouge2'],
    "meteor": meteor_score['meteor']
})

# Save updated JSON
with open('results/mscoco_phi4_updated_result.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
