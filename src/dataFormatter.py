from datasets import load_dataset
from dataProcessor import DataProcessor
from qagBase import QAGBase

class DataFormatter(QAGBase):
  def configure(self):
    self.dfCf = self.cp['dataFormatter']
    self.delim = self.dfCf['promptDelim']
    self.respTemple = self.dfCf[f'respTemple{self.trainFor}']
    self.fxMux()
    self.load()
  
  def fxMux(self):
    # multiplexing && setting functions
    format = self.dfCf['dataFormat']
    match format:
      case 'parHlSen_A': self.formatText = self.parHlSen_A
      case 'parHlAns_Q': self.formatText = self.parHlAns_Q
      case 'sen_As': self.formatText = self.sen_As
    
    if (self.cp['qagTrainer']['packing'] == 'True'):
      self.getExamples = self.formatText
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
  def unpackedProcessing(self, examples):
    output_texts = []
    # note that at this point, examples is a dictionary of lists. e.g.:
    # {'sentence': ['sent1', 'sent2' ...], 'answer': ['ans1', ...], ...}
    for i in range(len(examples["answer"])):
      text = self.formatText(examples, i)
      output_texts.append(text)
    return output_texts

  def sampleDataInput(self, i = 0):
    return self.formatText([self.trainDataset][0], i)
  
  def getEvalInputs(self, numInputs):
    inputs = []
    for i in range(numInputs):
      inputs.append(self.sampleDataInput(i))
    return inputs
  
  def getEvalSample(self): # update when we have better data resources
    dp = DataProcessor()    
    return (f'{self.delim} Highlighted context: {dp.getRandomVerse()}\n'
            + self.respTemple)

  # formatting f(x)s for input to various training phases
  def sen_As(self, example, i):
    return (f'{self.delim} Context: {example["sentence"][i]}'
            + f'{self.respTemple} {example["answer"][i]}')
  
  def parHlSen_A(self, example, i):
    return (f'{self.delim} Highlighted context: {example["paragraph_sentence"][i]}\n'
            + f'{self.respTemple} {example["answer"][i]}')

  def parHlAns_Q(self, example, i):
    return (f'{self.delim} Highlighted context: {example["paragraph_sentence"][i]} '
            + f'Answer: {example["answer"][i]}\n'
            + f'{self.respTemple} {example["question"][i]}')
  
if __name__ == '__main__':
  print('Testing DataProcessor . . .')
  df = DataFormatter()
  print(df.sampleDataInput())
