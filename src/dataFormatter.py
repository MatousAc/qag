from datasets import load_dataset
from configBase import ConfigBase
import random

class DataFormatter(ConfigBase):
  '''Handles training and inference data
  loading, processing, and formatting.'''
  def configure(self):
    self.dfCf = self.cp['dataFormatter']
    self.delim = self.dfCf['delim']
    self.respTemple = self.dfCf[f'respTemple{self.trainFor}']
    self.respKey = self.dfCf[f'respKey{self.trainFor}']
    self.load()
    if (self.cp['train']['packing'] == 'True'):
      self.getExamples = self.formatInput
    else: self.getExamples = self.unpackedProcessing

  def load(self):
    '''loads dataset'''
    if not self.quiet: print('Loading Data . . .')
    dsDict = load_dataset(self.paths['data'])
    # filter by quality
    threshold = int(self.dfCf['qualityThreshold'])
    dsDict = dsDict.filter(lambda row: row['quality'] >= threshold)

    if len(dsDict) == 1:
      if not self.quiet: print('Splitting data . . .')
      key = [split for split in dsDict][0]
      evalToTrainRatio = 0.001 if self.mode == 'test' else float(self.dfCf['evalToTrainRatio'])
      dsDict = dsDict[key].train_test_split(test_size=evalToTrainRatio)
    if not self.quiet:
      print('Results:')
      print(dsDict)
    
    self.trainDataset = dsDict['train']
    self.evalDataset = dsDict['test']

    if not self.quiet: print(f'''Loaded {len(self.trainDataset)} training and {len(self.evalDataset)} evaluation examples above at or above a quality of {threshold}''')

  def unpackedProcessing(self, examples):
    '''processes all data for training input'''
    output_texts = []
    # note that at this point, examples is a dictionary of lists. e.g.:
    # {'sentence': ['sent1', 'sent2' ...], 'answer': ['ans1', ...], ...}
    for i in range(len(examples["answer"])):
      text = self.formatInput(examples, i)
      output_texts.append(text)
    return output_texts
    
  def getInputSample(self, i = 0):
    return self.formatInput([self.trainDataset][0], i)

  ### formatting f(x)s for input to various training phases
  def formatInput(self, example, i = 0):
    '''Formats an example for training'''
    # get column values from lists of strings
    if isinstance(example['sentence'], list):
      sentence = example['sentence'][i]
      answer = example['answer'][i]
      if self.trainFor == 'QG': question = example['question'][i]
    else:
      sentence = example['sentence']
      answer = example['answer']
      if self.trainFor == 'QG': question = example['question']
    # construct example
    templ = self.respTemple
    templ = templ.replace('<context>', sentence)
    templ = templ.replace('<answer>', answer)
    if self.trainFor == 'QG': templ = templ.replace('<question>', question)
    return templ.strip()

  def getEvalInputs(self) -> tuple[list[str], list[str]]:
    '''Processes the evaluation dataset into a prompt for
    the model using the current training format and a label
    (desired output). Used in custom NLG metrics.'''
    inputs = []; labels = []
    for row in self.evalDataset:
      example = self.formatInput(row).split(self.respKey)
      inputs.append((example[0] + self.respKey).strip())
      labels.append(example[1].strip())
    return (inputs, labels)
  
  ## output
  def getInferenceInput(self, dp) -> str:
    '''Returns a prompt for inference'''
    sampleMode = self.dfCf['sampleMode']
    match sampleMode:
      case 'generate':
        input()
        templ = self.respTemple
      case 'manual':
        templ = input(f'> ')
        print('↓')
    if self.trainFor == 'AE':
      templ = templ.replace('<context>', dp.getRandomVerse().text)
      templ = templ.replace('<answer>', '')
    if self.trainFor == 'QG':
      row = random.randint(0, len(self.evalDataset) - 1)
      templ = templ.replace('<context>', self.evalDataset['sentence'][row])
      templ = templ.replace('<answer>', self.evalDataset['answer'][row])
      templ = templ.replace('<question>', '')
    return templ.strip()

if __name__ == '__main__':
  df = DataFormatter()
  df.printHeader('Testing DataFormatter')
  print(df.getInputSample())
