import evaluate
import numpy as np
from transformers import AutoTokenizer, EvalPrediction 


  # eval_pred:  <transformers.trainer_utils.EvalPrediction object at 0x0000018ABBC6ED10>
  
  # predictions:  [[    0   363   410 ...     0     0     0]
  #  [    0   363    19 ...     0     0     0]
  #  [    0   363    19 ...     0     0     0]
  #  ...
  #  [    0  2645   410 ...     0     0     0]
  #  [    0   363   410 ...     9    23     9]
  #  [    0   363   410 ... 26783     1     0]]
  # labels:  [[ 125   47   16 ... -100 -100 -100]
  #  [ 363  405 4173 ... -100 -100 -100]
  #  [ 363   19   45 ... -100 -100 -100]
  #  ...
  #  [2645 1622   12 ... -100 -100 -100]
  #  [   3 2544  520 ... -100 -100 -100]
  #  [ 363  410  717 ... -100 -100 -100]]

tokenizer = AutoTokenizer.from_pretrained('/home/ac/code/qag/models/llama-hf/7b')
tokenizer.pad_token = tokenizer.eos_token
metricFx = evaluate.load('google_bleu')

def detokenize(tokens):
  print('Detokenizing Tokens:')
  print(type(tokens))
  print(tokens)
  return tokenizer.decode(tokens, skip_special_tokens=True)

def detokenizeNdArr(arr):
  stringArr = []
  for intList in arr:
    stringArr.append(detokenize(intList))
  return stringArr

def computeMetric(evalPred): # currently broken
  # predictions = ["hello there general kenobi", "foo bar foobar"]
  # references = [["hello there general kenobi", "hello there !"],["foo bar foobar"]]
  # bleu = evaluate.load("bleu")
  # results = bleu.compute(predictions=predictions, references=references)
  # print(results)
  
  preds, labels = evalPred
  # do we have to decode our predictions?
  decodedPreds = detokenizeNdArr(preds)
  
  # replace -100 in the labels as we can't decode them.
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decodedLabels = detokenizeNdArr(labels)
  decodedLabels = [[l] for l in decodedLabels]
  
  # why can't I see anything I print out? I can't even log from here
  print(f'Evaluating with Gooogle Bleu')
  print('Predictions:')
  print(preds)
  print(decodedPreds)
  print('References:')
  print(labels)
  print(decodedLabels)
  
  # for ref, pred in zip(decoded_preds, decoded_labels):
  #   metricFx.add(references=ref, predictions=pred)
  # metrics = metricFx.compute()
  # takes: predictions (list of str): list of translations to score.
  #        references (list of list of str): list of lists of references for each translation.
  metrics = metricFx.compute(predictions=decodedPreds, references=decodedLabels)
  print(metrics)
  return metrics

ev = np.array([
  [0,   363,  410,  0,    12,   0     ],
  [0,   363,  19,   0,    1231, 0     ],
  [123, 363,  19,   1345, 0,    23212 ],
  [643, 2645, 410,  1252, 123,  0     ],
  [234, 363,  410,  9,    23,   9     ],
  [0,   363,  410,  26783,1,    0     ]
])

pred = np.array([
  [0,   630,  410,  0,    12,   -100  ],
  [2,   343,  19,   20,    123, -100  ],
  [123, 363,  19,   1345, -100,  2212 ],
  [643, 2645, 410,  1252, 123,  -100  ],
  [234, 363,  420,  9,    23,   -100  ],
  [0,   362,  410,  26783,1,    -100  ]
])

evalPred = EvalPrediction(ev,pred)

print(evalPred)

computeMetric(evalPred)
