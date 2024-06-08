# import libraries we need
import torch, sys, os, pandas as pd, re
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from datasets import load_dataset, DatasetDict
from modelHandler import ModelHandler
from verse import Verse

class Generator(ModelHandler):
  '''Handles QAG in a pipelined fashion using potentially different
  adapters. Handles model output post processing. Also handles final
  model evaluation.'''
  def startup(self):
    self.defineLists()
    self.oldModels = False
    self.interactive = False
    self.timer.mode = 'norm'
    self.pipelineFolders = {
      'AE' : '',
      'QG' : ''
    }

  def defineLists(self):
    self.diverseList = [
      'Luke 2:1', 'Ephesians 2:8-9', 'Judges 6:11', 
      'Genesis 6:1', 'Exodus 14:5', 'Joshua 1:1', '3 John 1:14', 
      'Judges 6:1', 'Judges 12:8', 'Malachi 4:2', 'Jonah 3:4', 
      'Jonah 1:13', 'Joshua 18:14', 'Philemon 1:12', 'Ezekiel 12:6', 
      'Ezekiel 12:1', 'Ezekiel 12:12', 'Galatians 1:1', 
      'Colossians 1:1', 'Psalm 20:4', 'Proverbs 1:7', 
      'Proverbs 15:1', 'Micah 5:1', 'Micah 5:2', 'Exodus 26:5', 
      'Exodus 25:13', 'Genesis 6:13', 'Genesis 6:2'
    ]

  def loadPipeline(self):
    self.bnbConfig = BitsAndBytesConfig(
      load_in_4bit = True,
      bnb_4bit_quant_type = 'nf4',
      bnb_4bit_compute_dtype = torch.float16
    )
    # load base model and tokenizer
    self.model = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path = self.paths['base'],
      quantization_config = self.bnbConfig,
      device_map = 'auto'
    )
    # assume same tokenizer for all adapters
    self.tokenizer = AutoTokenizer.from_pretrained(self.paths['base'])
    self.tokenizer.pad_token = self.tokenizer.eos_token
    # load adapters
    self.loadLora('AE')
    self.loadLora('QG')
    
    # setup generation destination
    aeStr = self.pipelineFolders['AE'][-5:-1] # get AE##
    qgStr = self.pipelineFolders['QG'][-5:-1] # get QG##
    self.qaOutputDir = self.basePath + f"/data/gen/qag{aeStr}{qgStr}"

  def loadLora(self, pipelineType: str = None):
    if not pipelineType: pipelineType = self.trainFor
    if self.oldModels: ending = input(f'{pipelineType} model number: ')
    else: ending = str(self.getLatestModelNumber(pipelineType))
    ending = ending.zfill(2)
    modeFolder = f'{self.basePath}/models/output/norm/' # mode folder
    modelFolder = f'{self.modelSize}b-{self.modelType}{pipelineType}{ending}/' # folder
    checkpointLocation = self.getLatestCheckpointPath(modeFolder + modelFolder)
    self.pipelineFolders[pipelineType] = modelFolder
    if not self.quiet: print(f'Loading {pipelineType} model from {modelFolder}')
    # load and name adapters for later use individually
    # merging adapters results in poor performance
    _ = self.model.load_adapter(checkpointLocation, adapter_name=pipelineType)

  def infer(self, inferenceInput: str, pipelineType: str):
    self.timer.model = self.pipelineFolders[pipelineType]
    self.timer.start()
    self.model.set_adapter(pipelineType)
    modelInput = self.tokenizer(inferenceInput, return_tensors='pt').to('cuda')
    self.model.eval()
    with torch.no_grad():
      tokens = self.model.generate(**modelInput, max_new_tokens=100)[0]
      output = self.tokenizer.decode(tokens, skip_special_tokens=True)
      # print(output)
      self.timer.stop() # the model's job is done @ this point
      # only return what was generated
      response = output.split(self.cp['dataFormatter'][f'respTemple{pipelineType}'])[1]
      return response

  def verseToQA(self, verse: Verse) -> pd.DataFrame:
    qa = pd.DataFrame(columns=['question', 'answer'])
    def countPoints(answer: str):
      numRe = r'\((\d+)\)'
      return max(len(re.findall(numRe, answer)), 1)
    # AE
    aeInput = self.df.formatInput({'sentence': verse.text, 'answer': ''}, formatFor = 'AE')
    # split answers and ditch the last one
    answers = self.infer(aeInput, 'AE').split('<sep>')[:-1]
    answers = [a.strip() for a in answers] # clean whitespace
    answers = self.dp.aeFilter(answers)
    answers = self.dp.aeDeduplicate(answers)
    if not self.quiet: print(answers)
    # QG
    for answer in answers:
      qgInput = self.cp['dataFormatter'][f'inputTempleQG']
      qgInput = self.df.formatInput({
        'sentence': verse.questionContext, 'answer': answer, 'question': verse.ref + ','
      }, formatFor = 'QG')
      question = self.infer(qgInput, 'QG')
      question = question.split('?')[0] # only the first question is relevant
      ptNum = countPoints(answer) # count points and prepend
      question = question.strip()
      question = f'({ptNum}pt{"s" if ptNum > 1 else ""}) {question}?'
      # FIXME czech for cut off quotes here and add them
      qa.loc[len(qa)] = [question, answer]
    if not self.quiet: print(qa)
    return qa

  def genMux(self, ref = None, verse = None):
        if verse == None: verse = self.requestVerse(ref)
        if not self.quiet: print(verse.text)
        return self.verseToQA(verse)

  def bibleToQAFiles(self):
    numRefs = len([vs for vsList in self.refList for vs in vsList])
    i = 0
    self.printProgressBar(i, numRefs, label = f'QAG')
    for vsList in self.refList:
      fileName = vsList[0].split(':')[0] # get filename
      filepath = os.path.normpath(f'{self.qaOutputDir}/{fileName}.csv')
      os.makedirs(os.path.dirname(filepath), exist_ok=True)
      file = open(filepath, 'w')
      qa = pd.DataFrame(columns = ['question', 'answer'])
      for ref in vsList:
        verse = self.dp.constructVerse(ref)
        qa = pd.concat([qa, self.genMux(verse = verse)])
        i += 1
        self.printProgressBar(i, numRefs, label = f'QAG now @ {fileName}')
      qa.to_csv(file, index=False)

  def interactiveGen(self):
    print('Ctrl+C to exit')
    try:
      if self.refList:
        for ref in self.refList: self.genMux(ref)
      else:
        while True: self.genMux()
    except KeyboardInterrupt: print(f'\rClosing{" " * 20}\n')
    except: raise

  def evalFileGen(self):
    print('Ctrl+C to exit')
    qLim = input('Enter file question limit as a number. Enter for no limit: ')
    numFiles = input('How many files do you want to generate. Enter for 1: ')
    if qLim == '': qLim = None
    else: qLim = int(qLim)
    if numFiles == '': numFiles = 1
    else: numFiles = int(numFiles)
    
    if qLim: self.printProgressBar(0, qLim * numFiles, label = 'generating questions')
    for fileNum in range(numFiles):
      filepath = os.path.normpath(f'{self.qaOutputDir}/pbeQA{fileNum}.csv')
      os.makedirs(os.path.dirname(filepath), exist_ok=True)
      file = open(filepath, 'w')
      cols = ['reference', 'additionalContext', 'verse', 'question', 'answer', 'grammaticality', 'acceptability']
      qa = pd.DataFrame(columns = cols)
      while len(qa) < qLim:
        verse = self.dp.getRandomVerse()
        currQA = self.genMux(verse = verse)
        currQA['reference'] = verse.ref
        currQA['additionalContext'] = verse.inContext
        currQA['verse'] = verse.text
        currQA['grammaticality'] = ''
        currQA['acceptability'] = ''
        currQA = currQA[cols]
        qa = pd.concat([qa, currQA])
        if qLim: self.printProgressBar(len(qa) + (qLim * fileNum), qLim * numFiles, label = 'generating questions')
      qa.to_csv(file, index=False)

  def requestVerse(self, ref = None) -> Verse:
    # get reference from user
    if not ref: ref = input('Reference: ')
  
    print(f'Reference: {ref}')
    if ref != '':
      try: return self.dp.constructVerse(ref)
      except IndexError:
        print('\rInvalid reference. Try again.')
        return self.requestVerse() # retry
      except: raise
    else: # grab random verse
      return self.dp.getRandomVerse()

  def autoEval(self):
    data = load_dataset(self.paths['data'])['train']
    def rowToVerse(row: dict):
      return self.dp.constructVerse(row['book'], row['chapter'], row['verse'], row['endVerse'])
    # split on counts
    dsDict = DatasetDict()
    for i in range(1,5):
      dsDict[str(i)] = data.filter(lambda row: row['count'] == i)
    dsDict['5+'] = data.filter(lambda row: row['count'] >= 5)

    logFile = open(os.path.normpath(self.basePath + '/data/logs/autoEval.txt'), 'w')
    for name, dataset in dsDict.items():
      verses = [rowToVerse(row) for row in dataset]
      labels = dataset['qa']
      preds = []
      for i, v in enumerate(verses):
        df = self.verseToQA(v)
        df['qa'] = 'Question: ' + df['question'] + '\nAnswer: ' + df['answer']
        preds.append(df['qa'].str.cat(sep = '\n'))
        self.printProgressBar(i, len(verses), label = f'gen predicitons for {name}')
      self.printProgressBar(len(verses), len(verses), label = f'gen predicitons for {name}')
      metrics = self.calculateMTMetrics(preds = preds, labels = labels)
      logFile.write(f'Dataset QA count: {name}\nNumber of rows: {dataset.num_rows}\n')
      logFile.write(str(metrics) + '\n')
      logFile.flush() # force write as we go along so we log progress

if __name__ == '__main__':
  generator = Generator()
  args = [arg.lower().replace('-', '') for arg in sys.argv]
  
  if 'oldmodels' in args: generator.oldModels = True
  else: generator.oldModels = False
  generator.loadPipeline()
  # optional pre-determined source
  if 'diverselist' in args: generator.refList = generator.diverseList
  elif 'evallist' in args: generator.refList = generator.diverseList
  else: generator.refList = None
  if 'fromreference' in args:
    texts = input("Enter references for qag in a comma-separated list: ")
    texts = texts.split(', ')
    generator.refList = generator.dp.enumerateContext(texts)
  
  # main function execution
  if 'interactive' in args: generator.interactiveGen()
  elif 'autoeval' in args: generator.autoEval()
  elif 'manualeval' in args: generator.evalFileGen()
  elif 'fromreference' in args: generator.bibleToQAFiles()
  else: generator.interactiveGen()
