from datasets import load_dataset
import configparser
from dataProcessor import DataProcessor

class DataFormatter():
  def __init__(self, configFilePath = 'qag.ini'):
    config = configparser.ConfigParser()
    config.read(configFilePath)
    self.paths = config['paths']
    self.genCf = config['general']
    self.dfCf = config['dataFormatter']
    self.quiet = self.genCf['quiet'] == 'True'
    self.delim = self.dfCf['promptDelim']

    # setting functions
    format = self.dfCf['dataFormat']
    match format:
      case 'parHlSen_A': self.formatText = self.parHlSen_A
      case 'parHlAns_Q': self.formatText = self.parHlAns_Q
      case 'sen_As': self.formatText = self.sen_As
    
    if (config['qagTrainer']['packing'] == 'True'):
      self.getExamples = self.formatText
    else: self.getExamples = self.unpackedProcessing

    self.load()
  
  def load(self):
    if not self.quiet: print('Loading Data . . .')
    dsDict = load_dataset(self.paths['data'])
    
    if 'AE' in self.paths['data']:
      for split, dataset in dsDict.items():
        dsDict[split] = dataset.filter(
          lambda row: row['count'] > int(self.dfCf['aeMinAnswerCount'])
        )
    
    if not self.quiet: print(dsDict)
    if len(dsDict) == 1:
      key = [split for split in dsDict][0]
      dsDict = dsDict[key].train_test_split(test_size=float(self.dfCf['evalToTrainRatio']))
    if not self.quiet: print(dsDict)
    
    self.trainDataset = dsDict['train']
    self.evalDataset = dsDict['test']

    if not self.quiet: print(self.trainDataset[0])


  # data processing f(x)s
  def unpackedProcessing(self, examples):
    output_texts = []
    for i in range(len(examples["answer"])):
      text = self.parHlSen_A(examples[i])
      output_texts.append(text)
    return output_texts

  def sampleDataInput(self, i = 0):
    print(self.formatText([self.trainDataset[i]][0]))
  
  def getEvalSample(self): # update when we have better data resources
    dp = DataProcessor()    
    return (f'{self.delim} Highlighted context: {dp.getRandomVerse()}\n '
            + f'{self.delim} Answer: ')

  # formatting f(x)s for input to various training phases
  def sen_As(self, example):
    return (f'{self.delim} Context: {example["sentence"]}\n '
            + f'{self.delim} Key nouns, actions, and phrases: {example["answer"]}')
  
  def parHlSen_A(self, example):
    return (f'{self.delim} Highlighted context: {example["paragraph_sentence"]}\n '
            + f'{self.delim} Answer: {example["answer"]}')

  def parHlAns_Q(self, example):
    return (f'{self.delim} Highlighted context: {example["paragraph_sentence"]} '
            + f'Answer: {example["answer"]}\n '
            + f'{self.delim} Question {example["question"]}')
  
if __name__ == '__main__':
  print('Testing DataProcessor . . .')
  df = DataFormatter()
  df.sampleDataInput()
