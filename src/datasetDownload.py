from datasets import load_dataset, concatenate_datasets, DatasetDict

source = input('hf dataset name: ')
source = source if source else 'lmqg/qg_squad'
destination = f'/home/ac/code/qag/data/{input("save to ./data/...: ")}'
doConcat = input('Concatenate splits? (y/n): ') == 'y'

datasetDict = load_dataset(source)
print(datasetDict)

if doConcat:
  datasetDict = concatenate_datasets([d for k, d in datasetDict.items()])
  datasetDict = DatasetDict({'train': datasetDict})
  print(datasetDict)


for split, dataset in datasetDict.items():
    dataset.to_json(f"{destination}/{split}.jsonl")
