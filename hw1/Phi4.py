# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)
