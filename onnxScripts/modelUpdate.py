import json
import shutil

src = './onnx/optimum'
dest = './pbe/src/assets/OptimumONNX'
# rm dest
shutil.rmtree(dest, ignore_errors=True)
# copy
shutil.copytree(src, dest)

# create a tokenizer map
with open(dest + '/tokenizer.json', 'r', encoding='UTF-8') as f:
  # Load the contents of the file into a Python object
  tokenizer = json.load(f)

vocab = tokenizer['model']['vocab']

tokenMap = {}
for i, pair in enumerate(vocab):
  key = pair[0]
  if key[0] == '▁' and key != '▁':
    key = key[1:]
  if key not in tokenMap:
    tokenMap[key] = i

with open(dest + '/tokenMap.json', 'w', encoding='UTF-8') as f:
  json.dump(tokenMap, f, ensure_ascii=False)
