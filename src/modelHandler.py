import torch, os, evaluate
from configBase import ConfigBase
from dataFormatter import DataFormatter
from dataProcessor import DataProcessor
from timeLogger import TimeLogger
from gstop import GenerationStopper, STOP_TOKENS_REGISTRY
from mt import MT

class ModelHandler(ConfigBase):
  '''A base class that sets up common settings and executes
  common functionality between the Trainer and the Generator.'''
  def configure(self):
    # increment output folder number
    self.genCf = self.cp['generate']
    self.oldModels = False
    
    STOP_TOKENS_REGISTRY['llama-2'] = { '?': [29973] }
    STOP_TOKENS_REGISTRY['llama-3'] = { '?': [30] }
    modelVersion = 'llama-2' if self.modelSize == '7' else 'llama-3'
    self.genStopper = GenerationStopper(STOP_TOKENS_REGISTRY[modelVersion])
    
    latestModelNum = self.getLatestModelNumber()
    self.outputDir = self.paths['output'] + str(latestModelNum + 1).zfill(2)
    self.latestModelDir = self.paths['output'] + str(latestModelNum).zfill(2)
    self.df = DataFormatter()
    self.dp = DataProcessor()
    self.timer = TimeLogger()
    self.timer.mode = 'norm'
    
    self.modelFolders = {t: '' for t in MT}
    self.startup()

  def startup(self):
    '''Configuration for further derived classes'''
    pass

  def getLatestModelNumber(self, pipelineType: MT|None = None):
    '''Returns the latest AE/QG/E2E model (defaults to self.type) in
    the models/mode directory if one is present. Else -1'''
    if not pipelineType: pipelineType = self.type
    parent = self.paths['output'][:self.paths['output'].find(self.mode)] + self.mode
    subfolders = [f.name for f in os.scandir(parent) if f.is_dir() and pipelineType.value in f.name]
    subfolderNumbers = [int(f[-2:]) for f in subfolders]
    return max(subfolderNumbers) if len(subfolders) else -1
  
  def getLatestCheckpointPath(self, modelDir):
    '''Returns the path of the latest checkpoint in the 
    model directory passed, or false if DNE.'''
    prefix = 'checkpoint-'
    subfolders = [f.name for f in os.scandir(modelDir) if f.is_dir()]
    if len(subfolders) == 0: return False
    subfolderNumbers = [int(f.replace(prefix, '')) for f in subfolders]
    return os.path.normpath(f'{modelDir}/{prefix}{max(subfolderNumbers)}')


  def loadLora(self, pipelineType: MT|None = None):
    if not pipelineType: pipelineType = self.type
    if self.oldModels: ending = input(f'{pipelineType.value} model number: ')
    else: ending = str(self.getLatestModelNumber(pipelineType))
    ending = ending.zfill(2)
    modeFolder = f'{self.basePath}/models/output/norm/' # mode folder
    modelFolder = f'{self.modelSize}b-{self.baseType}{pipelineType.value}{ending}/' # folder
    checkpointLocation = self.getLatestCheckpointPath(modeFolder + modelFolder)
    self.modelFolders[pipelineType.value] = modelFolder
    if not self.quiet: print(f'Loading {pipelineType.value} model from {modelFolder}')
    # load and name adapters for later use individually
    # merging adapters results in poor performance
    _ = self.model.load_adapter(checkpointLocation, adapter_name=pipelineType.value)


  def infer(self, inferenceInput: str, pipelineType: MT|None = None, reduce = True):
    ''' Inference using adapter of pipelineType (defaults to self.type).
    Optionally reduces output to only what was generated.'''
    if not pipelineType: pipelineType = self.type
    self.timer.model = self.modelFolders[pipelineType]
    self.timer.start()
    self.model.set_adapter(pipelineType.value)
    modelInput = self.tokenizer(inferenceInput, return_tensors='pt').to('cuda')
    self.model.eval()
    with torch.no_grad():
      tokens = self.model.generate(
        **modelInput,
        max_new_tokens = int(self.genCf['maxLength']),
        repetition_penalty = float(self.genCf['repetitionPenalty']),
        stopping_criteria = self.genStopper.criteria if pipelineType == MT.QG else None
      )[0]
      output = self.tokenizer.decode(tokens, skip_special_tokens=True)
      # print(output)
      self.timer.stop() # the model's job is done @ this point
      # only return what was generated
      if reduce: output = output.split(self.df.dfCf[f'respTemple{pipelineType.value}'])[1]
      return output


  def calculateMTMetrics(self, preds: list[str], labels: list[str]) -> dict:
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    # takes: predictions (list of str): translations to score.
    #        references (list of list of str|list of str): references for each translation.
    result = {
      'bleu4': bleu.compute(predictions=preds, references=labels)['precisions'][3], # to get bleu-4
      'rogueL': rouge.compute(predictions=preds, references=labels, 
                use_stemmer=True, use_aggregator=True, rouge_types=['rougeL'])['rougeL'],
      'meteor': meteor.compute(predictions=preds, references=labels)['meteor']
    }
    result = {key: value * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()} # round for 4 decimal places
    return result

if __name__ == '__main__':
  ModelHandler()
  print("No news is good news.")