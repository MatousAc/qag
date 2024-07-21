import os, random
from datasets import load_dataset, Dataset
from configparser import ConfigParser, ExtendedInterpolation
from configBase import ConfigBase
from mt import MT

class DataFormatter(ConfigBase):
  '''Handles training and inference data
  loading, processing, and formatting.'''
  def configure(self, templates = '/src/inputTemplates.ini'):
    # config 1. get input templates 2. get other settings
    self.dfCp = ConfigParser(interpolation=ExtendedInterpolation())
    self.dfCp.read(os.path.normpath(self.basePath + templates))
    self.dfCf = self.dfCp['templates']
    # combine settings from two .ini files
    for entry in self.cp['data']: self.dfCf[entry] = self.cp['data'][entry]
    self.delim = self.dfCf['delim']
    self.inputTemple = self.dfCf[f'inputTemple{self.type.value}']
    self.respTemple = self.dfCf[f'respTemple{self.type.value}']
    self.load()
    if (self.cp['train']['packing'] == 'True'):
      self.getExamples = self.formatInput
    else: self.getExamples = self.unpackedProcessing

  def load(self, threshold: int = None, shuffle = True):
    '''loads dataset'''
    if not self.quiet: self.printHeader('Loading Data')
    dsDict = load_dataset(self.paths['data'])
    # filter by quality
    if 'eval' in self.paths['data']: return # no need to load here
    if not threshold: threshold = int(self.dfCf['qualityThreshold'])
    dsDict = dsDict.filter(lambda row: float(row['quality']) >= threshold)
    if shuffle: dsDict = dsDict.shuffle(seed=42) # 42. why not?

    if len(dsDict) == 1:
      key = [split for split in dsDict][0]
      evalToTrainRatio = 0.001 if self.mode == 'test' else float(self.dfCf['evalToTrainRatio'])
      dsDict = dsDict[key].train_test_split(test_size = evalToTrainRatio)
    if not self.quiet:
      print('Results:')
      print(dsDict)
    
    self.trainDataset = dsDict['train']
    self.evalDataset = dsDict['test']

    if not self.quiet: print(f'''Loaded {len(self.trainDataset)} training and {len(self.evalDataset)} evaluation examples >= a quality of {threshold}''')

  def unpackedProcessing(self, examples) -> list:
    '''processes all data for training input'''
    output_texts = []
    # note that at this point, examples is a dictionary of lists. e.g.:
    # {'sentence': ['sent1', 'sent2' ...], 'answer': ['ans1', ...], ...}
    for i in range(len(examples['sentence'])):
      text = self.formatInput(examples, i)
      output_texts.append(text)
    return output_texts
    
  def getInputSample(self, i = 0):
    return self.formatInput([self.trainDataset][0], i)

  ### formatting f(x)s for input to various training phases
  def formatInput(self, example, i = 0,  formatFor: MT|None = None) -> str:
    '''Formats an example for training or for generation'''
    # make access easier below:
    if isinstance(example['sentence'], list): isList = True
    else: isList = False
    def get(col):
      if isList: return example[col][i] # normal data processing
      else: return example[col] # for packed processing or generation
    if formatFor == None: formatFor = self.type # default
    # actually format stuff:
    templ = self.dfCf[f'inputTemple{formatFor.value}']
    if formatFor != MT.E2E:
      templ = templ.replace('<answer>', get('answer'))
      templ = templ.replace('<context>', get('sentence'))
    if formatFor == MT.QA:
      ans = get('answer')
      templ = templ.replace('<answer>', ans + self.sep + ans)
    if formatFor == MT.QG or formatFor == MT.QA:
      templ = templ.replace('<question>', get('question'))
    if formatFor == MT.E2E:
      templ = templ.replace('<context>', f"{get('ref')} - {get('sentence')}")
      templ = templ.replace('<qa>', get('qa'))
    return templ.strip()

  def getEvalInputs(self, evalDataset: Dataset = None) -> tuple[list[str], list[str]]:
    '''Processes the evaluation dataset into a prompt for
    the model using the current training format and a label
    (desired output). Used in custom NLG metrics.'''
    if evalDataset == None: evalDataset = self.evalDataset
    inputs = []; labels = []
    for row in evalDataset:
      example = self.formatInput(row).split(self.respTemple)
      inputs.append((example[0] + self.respTemple).strip())
      labels.append(example[1].strip())
    return (inputs, labels)
  
  ## output
  def getInferenceInput(self, dp) -> str:
    '''Returns a prompt for inference'''
    sampleMode = self.dfCf['sampleMode']
    match sampleMode:
      case 'generate':
        input()
        templ = self.inputTemple
      case 'manual':
        templ = input(f'> ')
        print('↓')
    if self.type == MT.AE:
      templ = templ.replace('<context>', dp.getRandomVerse().text)
      templ = templ.replace('<answer>', '')
    if self.type == MT.QG:
      row = random.randint(0, len(self.evalDataset) - 1)
      templ = templ.replace('<context>', self.evalDataset['sentence'][row])
      templ = templ.replace('<answer>', self.evalDataset['answer'][row])
      templ = templ.replace('<question>', '')
    if self.type == MT.QA:
      row = random.randint(0, len(self.evalDataset) - 1)
      templ = templ.replace('<context>', self.evalDataset['sentence'][row])
      templ = templ.replace('<question>', self.evalDataset['question'][row])
      templ = templ.replace('<answer>', '')
    if self.type == MT.E2E:
      verse = dp.getRandomVerse()
      templ = templ.replace('<context>', f'{verse.ref} - {verse.text}')
      templ = templ.replace('<qa>', '')
    return templ.strip()

if __name__ == '__main__':
  df = DataFormatter()
  df.printHeader('Testing DataFormatter')
  print(df.getInputSample())
