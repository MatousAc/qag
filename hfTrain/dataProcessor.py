from datasets import load_dataset
import configparser

class DataProcessor():
  def __init__(self, configFilePath = 'QAG.ini'):
    config = configparser.ConfigParser()
    config.read(configFilePath)
    self.config = config['data']
    self.load()
  
  def load(self, quiet = False):
    data = load_dataset('lmqg/qg_squad')
    if not quiet: print(data)

    self.train_dataset = data['train']
    self.eval_dataset = data['validation']

    if not quiet: print(self.train_dataset[0])


  # define data processing functions that produce the actual untokenized input for various training phases
  def contextAnswer(self, example, i):
    return f'Select answer: {example["paragraph_sentence"][i]}\n Answer: {example["answer"][i]}'

  def answer(self, example):
    return f'Answer: {example["answer"]}'

  def processData(self, examples):
    output_texts = []
    for i in range(len(examples["answer"])):
      text = self.contextAnswer(examples, i)
      output_texts.append(text)
    return output_texts
