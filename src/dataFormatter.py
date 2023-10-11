from datasets import load_dataset
from dataProcessor import DataProcessor
from qagBase import QAGBase

class DataFormatter(QAGBase):
  def configure(self):
    self.dfCf = self.cp['dataFormatter']
    self.delim = self.dfCf['promptDelim']
    self.responseTemplate = f' {self.dfCf["responseTemplate"]}'

    # setting functions
    format = self.dfCf['dataFormat']
    match format:
      case 'parHlSen_A': self.formatText = self.parHlSen_A
      case 'parHlAns_Q': self.formatText = self.parHlAns_Q
      case 'sen_As': self.formatText = self.sen_As
    
    if (self.cp['qagTrainer']['packing'] == 'True'):
      self.getExamples = self.formatText
    else: self.getExamples = self.unpackedProcessing

    self.load()
  
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
    for i in range(len(examples["answer"])):
      text = self.formatText(examples)
      output_texts.append(text)
    return output_texts

  def sampleDataInput(self, i = 0):
    print(self.formatText([self.trainDataset[i]][0]))
  
  def getEvalSample(self): # update when we have better data resources
    dp = DataProcessor()    
    return (f'{self.delim} Highlighted context: {dp.getRandomVerse()}\n'
            + self.responseTemplate)

  # formatting f(x)s for input to various training phases
  def sen_As(self, example):
    return (f'{self.delim} Context: {example["sentence"]}\n'
            + f'{self.responseTemplate} {example["answer"]}')
  
  def parHlSen_A(self, example):
    return (f'{self.delim} Highlighted context: {example["paragraph_sentence"]}\n'
            + f'{self.responseTemplate} {example["answer"]}')

  def parHlAns_Q(self, example):
    return (f'{self.delim} Highlighted context: {example["paragraph_sentence"]} '
            + f'Answer: {example["answer"]}\n'
            + f'{self.responseTemplate  } {example["question"]}')
  
if __name__ == '__main__':
  print('Testing DataProcessor . . .')
  df = DataFormatter()
  df.sampleDataInput()
