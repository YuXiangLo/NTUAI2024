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

filename = 'results/flickr_blip_result.json'

# Load JSON results
with open(filename, 'r') as f:
    data = json.load(f)

# Extract and preprocess references and predictions
all_references = [[preprocess_text(ref) for ref in item['reference_captions']] for item in data['results']]
all_predictions = [preprocess_text(item['generated_caption']) for item in data['results']]

# Recompute metrics
bleu_score = bleu.compute(predictions=all_predictions, references=all_references)

rouge1_scores = 0
rouge2_scores = 0
meteor_score = 0
for i in range(len(all_references[0])):
    rouge1_scores += rouge.compute(predictions=all_predictions, references=[r[i] for r in all_references])['rouge1']
    rouge2_scores += rouge.compute(predictions=all_predictions, references=[r[i] for r in all_references])['rouge2']
    meteor_score += meteor.compute(predictions=all_predictions, references=[r[i] for r in all_references])['meteor']
rouge1_scores /= len(all_references[0])
rouge2_scores /= len(all_references[0])
meteor_score /= len(all_references[0])

# Print updated results
print(f"Corpus BLEU: {bleu_score['bleu']:.4f}")
print(f"ROUGE-1: {rouge1_scores:.4f}")
print(f"ROUGE-2: {rouge2_scores:.4f}")
print(f"METEOR: {meteor_score:.4f}")

# Update JSON with new results
data.update({
    "corpus_bleu": bleu_score['bleu'],
    "average_rouge1": rouge1_scores,
    "average_rouge2": rouge2_scores,
    "average_meteor": meteor_score
})

# Save updated JSON
with open(filename, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
