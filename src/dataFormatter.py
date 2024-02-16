from datasets import load_dataset
from qagBase import QAGBase

class DataFormatter(QAGBase):
  def configure(self):
    self.dfCf = self.cp['dataFormatter']
    self.delim = self.dfCf['delim']
    self.respTemple = self.dfCf[f'{self.modelType}RespTemple{self.trainFor}']
    self.respKey = self.dfCf[f'{self.modelType}RespKey{self.trainFor}']
    self.fxMux()
    self.load()
  
  def fxMux(self):
    # multiplexing && setting functions
    # format = self.dfCf['dataFormat']
    # match format:
    #   case 'parHlSen_A': self.formatText = self.parHlSen_A
    #   case 'parHlAns_Q': self.formatText = self.parHlAns_Q
    #   case 'sen_As': self.formatText = self.sen_As
    
    if (self.cp['qagTrainer']['packing'] == 'True'):
      self.getExamples = self.formatInput
    else: self.getExamples = self.unpackedProcessing
  
  def load(self):
    if not self.quiet: print('Loading Data . . .')
    dsDict = load_dataset(self.paths['data'])
    
    if 'AE' in self.paths['data']:
      for split, dataset in dsDict.items():
        dsDict[split] = dataset.filter(
          lambda row: row['count'] >= int(self.dfCf['aeMinAnswerCount'])
        )
    
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

  # data processing f(x)s
  ## input
  def unpackedProcessing(self, examples):
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
    '''returns input for training'''
    templ = self.respTemple
    templ = templ.replace('<context>', example["sentence"][i])
    templ = templ.replace('<answer>', example["answer"][i])
    if self.trainFor == 'QG': templ = templ.replace('<question>', example["question"][i])
    return templ.strip()

  # def sen_As(self, example, i):
  #   return (f'{self.delim} Context: {example["sentence"][i]}'
  #           + f'{self.respTemple} {example["answer"][i]}')
  
  # def parHlSen_A(self, example, i):
  #   return (f'{self.delim} Highlighted context: {example["paragraph_sentence"][i]}\n'
  #           + f'{self.respTemple} {example["answer"][i]}')

  # def parHlAns_Q(self, example, i):
  #   return (f'{self.delim} Highlighted context: {example["paragraph_sentence"][i]} '
  #           + f'Answer: {example["answer"][i]}\n'
  #           + f'{self.respTemple} {example["question"][i]}')
  
  ## output
  def getEvalInputs(self, numInputs):
    '''used in custom preprocessing of logits'''
    inputs = []
    for i in range(numInputs):
      inputs.append(self.getInputSample(i))
    return inputs
  
  def getInferenceInput(self, dp):
    '''returns a prompt for inference'''
    sampleMode = self.dfCf['sampleMode']
    match sampleMode:
      case 'generate':
        input()
        templ = self.dfCf[f'{self.modelType}RespTemple{self.trainFor}']
      case 'manual':
        templ = input(f'> ')
        print('↓')
    templ = templ.replace('<context>', dp.getRandomVerse())
    templ = templ.replace('<answer>', '')
    return templ.strip() #template.replace('<context>', dp.getRandomVerse())

if __name__ == '__main__':
  print('Testing DataFormatter . . .')
  df = DataFormatter()
  print(df.getInputSample())
