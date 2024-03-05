# import libraries we need
import torch, sys
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from qagBase import QAGBase
from dataFormatter import DataFormatter
from dataProcessor import DataProcessor
from timeLogger import TimeLogger
from verse import Verse

class Generator(QAGBase):
  def configure(self):
    self.modelCf = self.cp['model']
    self.oldModels = False
    self.timer = TimeLogger()
    self.timer.mode = 'norm'
    self.dp = DataProcessor()
    self.pipelineFolders = {
      'AE' : '',
      'QG' : ''
    }

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
    # load and name adapters for later use individuallly
    # merging adapters results in poor performance
    _ = self.model.load_adapter(checkpointLocation, adapter_name=pipelineType)

  def infer(self, inferenceInput: str, pipelineType: str):
    self.timer.start()
    self.model.set_adapter(pipelineType)
    modelInput = self.tokenizer(inferenceInput, return_tensors='pt').to('cuda')
    self.model.eval()
    with torch.no_grad():
      tokens = self.model.generate(**modelInput, max_new_tokens=100)[0]
      output = self.tokenizer.decode(tokens, skip_special_tokens=True)
      # print(output)
      self.timer.stop() # model work is done @ this point
      # only return what was generated
      response = output.split(self.cp['dataFormatter'][f'respKey{pipelineType}'])[1]
      return response

  def generateQA(self, verse: Verse):
    def smartUnCapitalize(str):
      if str.split()[0] in [
        'Who', 'What', 'When', 'Where', 'Why', 'Which', 
        'Whose','How', 'The', 'That', 'A'
      ]: str = str[0].lower() + str[1:]
      return str
    qa = []
    # AE
    aeInput = self.cp['dataFormatter'][f'respTempleAE']
    aeInput = aeInput.replace('<context>', verse.text)
    aeInput = aeInput.replace('<answer>', '')
    aeInput = aeInput.strip()
    # split answers and ditch the last one
    self.timer.model = self.pipelineFolders['AE']
    answers = self.infer(aeInput, 'AE').split('<sep>')[:-1]
    answers = [a.strip() for a in answers] # clean whitespace
    print(answers)
    answers = self.dp.aeDeduplicate(answers)
    # QG
    self.timer.model = self.pipelineFolders['QG']
    for answer in answers:
      qgInput = self.cp['dataFormatter'][f'respTempleQG']
      qgInput = qgInput.replace('<context>', verse.text)
      qgInput = qgInput.replace('<answer>', answer)
      qgInput = qgInput.replace('<question>', '')
      qgInput = qgInput.strip()
      question = self.infer(qgInput, 'QG')
      question = question.split('?')[0]
      question = question.strip()
      question = f'According to {verse.ref}, {smartUnCapitalize(question)}?'
      qa.append({
        'question': question,
        'answer': answer
      })
    if not self.quiet:
      for qaPair in qa:
        print(f'Question: ', qaPair['question'])
        print(f'Answer: ', qaPair['answer'])
    return qa

  def generationLoop(self):
    print('Ctrl+C to exit')
    try:
      while True:
        verse = self.requestVerse()
        print(verse.text)
        qa = self.generateQA(verse)
    except KeyboardInterrupt: print(f'\rClosing{" " * 20}\n')
    except: raise

  def requestVerse(self) -> Verse:
    # get reference from user
    ref = input('Reference: ')
    if ref != '':
      try: return self.dp.constructVerse(ref)
      except IndexError:
        print('\rInvalid reference. Try again.')
        return self.requestVerse() # retry
      except: raise
    else: # grab random verse
      return self.dp.getRandomVerse()

if __name__ == '__main__':
  df = DataFormatter()
  generator = Generator(dataFormatter=df)
  if len(sys.argv) == 1: cmd = '-latest'
  else: cmd = sys.argv[1]
  match cmd.replace('-', '').lower():
    case 'oldmodels': generator.oldModels = True
    case 'latest' | _: generator.oldModels = False
  generator.loadPipeline()
  generator.generationLoop()
