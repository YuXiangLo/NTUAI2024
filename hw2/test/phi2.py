import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure you're using the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Include attention mask
inputs = tokenizer(
    '''def print_prime(n):
   """
   Print all primes between 1 and n
   """''',
    return_tensors="pt",
    padding=True,
    return_attention_mask=True
)

# Move input tensors to the same device as model
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate
outputs = model.generate(**inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(text)
