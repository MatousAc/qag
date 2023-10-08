from datasets import load_dataset
import configparser

class DataProcessor():
  def __init__(self, configFilePath = 'qag.ini'):
    config = configparser.ConfigParser()
    config.read(configFilePath)
    self.paths = config['paths']
    self.config = config['general']
    self.quiet = bool(self.config['quiet'])

    # setting functions
    format = self.config['dataFormat']
    match format:
      case 'parHlSen_Ans': self.formatText = self.parHlSen_Ans
      case 'parHlAns_Q': self.formatText = self.parHlAns_Q
    
    if bool(self.config['packing']): self.getExamples = self.formatText
    else: self.getExamples = self.unpackedProcessing

    self.load()
  
  def load(self):
    print('##### Loading Data #####')
    data = load_dataset(self.paths['data'])
    if not self.quiet: print(data)

    self.train_dataset = data['train']
    self.eval_dataset = data['validation']

    if not self.quiet: print(self.train_dataset[0])


  # data processing f(x)s
  def unpackedProcessing(self, examples):
    output_texts = []
    for i in range(len(examples["answer"])):
      text = self.parHlSen_Ans(examples[i])
      output_texts.append(text)
    return output_texts

  def test(self, i = 0):
    print(self.formatText([self.train_dataset[i]]))

  # formatting f(x)s for input to various training phases
  def parHlSen_Ans(self, example):
    example = example[0]
    return (f'### Highlighted context: {example["paragraph_sentence"]}\n '
            + f'### Answer: {example["answer"]}')

  def parHlAns_Q(self, example):
    example = example[0]
    return (f'### Highlighted context: {example["paragraph_sentence"]} '
            + f'Answer: {example["answer"]}\n '
            + f'### Question {example["question"]}')
  
if __name__ == '__main__':
  print('##### Testing DataProcessor #####')
  dp = DataProcessor()
  dp.test()
