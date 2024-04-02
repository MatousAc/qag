import os, evaluate
from configBase import ConfigBase
from dataFormatter import DataFormatter
from dataProcessor import DataProcessor
from timeLogger import TimeLogger

class ModelHandler(ConfigBase):
  '''A base class that sets up common settings and executes
  common functionality between the Trainer and the Generator.'''
  def configure(self):
    # increment output folder number
    latestModelNum = self.getLatestModelNumber()
    self.outputDir = self.paths['output'] + str(latestModelNum + 1).zfill(2)
    self.latestModelDir = self.paths['output'] + str(latestModelNum).zfill(2)
    self.df = DataFormatter()
    self.dp = DataProcessor()
    self.timer = TimeLogger()
    self.startup()

  def startup(self):
    '''Configuration for further derived classes'''
    pass

  def getLatestModelNumber(self, pipelineType: str = None):
    '''Returns the latest AE/QG model (defaults to self.mode) in
    the models/mode directory if one is present. Else -1'''
    if not pipelineType: pipelineType = self.trainFor
    parent = self.paths['output'][:self.paths['output'].find(self.mode)] + self.mode
    subfolders = [f.name for f in os.scandir(parent) if f.is_dir() and pipelineType in f.name]
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

  def calculateMTMetrics(self, preds: list[str], labels: list[str]) -> dict:
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    # takes: predictions (list of str): translations to score.
    #        references (list of list of str|list of str): references for each translation.
    result = {
      'rogueL': rouge.compute(predictions=preds, references=labels, 
                use_stemmer=True, use_aggregator=True, rouge_types=['rougeL'])['rougeL'],
      'bleu': bleu.compute(predictions=preds, references=labels)['bleu'],
      'meteor': meteor.compute(predictions=preds, references=labels)['meteor']
    }
    result = {key: value * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()} # round for 4 decimal places
    return result

if __name__ == '__main__':
  ModelHandler()
  print("No news is good news.")