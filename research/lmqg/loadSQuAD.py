from datasets import load_dataset

# https://huggingface.co/datasets/lmqg/qg_squad
dataset = load_dataset("lmqg/qg_squad")
print(dataset)
dataset.head()