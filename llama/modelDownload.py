from transformers import AutoModelForCausalLM, AutoTokenizer

model_source = input('hf model name: ')
model_destination = f'/home/ac/code/qag/models/{input("save to (models/...): ")}/'

model = AutoModelForCausalLM.from_pretrained(model_source)
tokenizer = AutoTokenizer.from_pretrained(model_source)
model.save_pretrained(model_destination)
tokenizer.save_pretrained(model_destination)
