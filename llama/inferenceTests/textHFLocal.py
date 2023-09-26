# Load model directly
from transformers import AutoTokenizer
import transformers
import torch

model = "../models/llama-hf/7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
  "text-generation",
  model=model,
  torch_dtype=torch.float16,
  device_map="auto",
)

sequences = pipeline(
  'When in the course of human events, it',
  do_sample=True,
  top_k=10,
  num_return_sequences=1,
  eos_token_id=tokenizer.eos_token_id,
  max_length=200,
)

for seq in sequences:
  print(f"Result: {seq['generated_text']}")