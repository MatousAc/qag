from datasets import load_dataset
from qagBase import QAGBase
import random

class DataFormatter(QAGBase):
  def configure(self):
    self.dfCf = self.cp['dataFormatter']
    self.delim = self.dfCf['delim']
    self.respTemple = self.dfCf[f'{self.modelType}RespTemple{self.trainFor}']
    self.respKey = self.dfCf[f'{self.modelType}RespKey{self.trainFor}']
    self.load()
    if (self.cp['model']['packing'] == 'True'):
      self.getExamples = self.formatInput
    else: self.getExamples = self.unpackedProcessing

  def load(self):
    '''loads dataset'''
    if not self.quiet: print('Loading Data . . .')
    dsDict = load_dataset(self.paths['data'])
    
    if len(dsDict) == 1:
      if not self.quiet: print('Splitting data . . .')
      key = [split for split in dsDict][0]
      dsDict = dsDict[key].train_test_split(test_size=float(self.dfCf['evalToTrainRatio']))
    if not self.quiet:
      print('Results:')
      print(dsDict)
    
    self.trainDataset = dsDict['train']
    self.evalDataset = dsDict['test']

    if not self.quiet: print(self.trainDataset[0])

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
    '''Returns input for training'''
    templ = self.respTemple
    templ = templ.replace('<context>', example["sentence"][i])
    templ = templ.replace('<answer>', example["answer"][i])
    if self.trainFor == 'QG': templ = templ.replace('<question>', example["question"][i])
    return templ.strip()

  ## output
  def getEvalInputs(self, numInputs):
    '''Used in custom preprocessing of logits'''
    inputs = []
    for i in range(numInputs):
      inputs.append(self.getInputSample(i))
    return inputs
  
  def getInferenceInput(self, dp):
    '''Returns a prompt for inference'''
    sampleMode = self.dfCf['sampleMode']
    match sampleMode:
      case 'generate':
        input()
        templ = self.dfCf[f'{self.modelType}RespTemple{self.trainFor}']
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
