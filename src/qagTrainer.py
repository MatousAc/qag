# import libraries we need
import configparser
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments 
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from peft import PeftModel
from dataFormatter import DataFormatter

class QAGTrainer:
  def __init__(self, configFilePath = 'qag.ini'):
    config = configparser.ConfigParser()
    config.read(configFilePath)
    self.paths = config['paths']
    self.peft = config['peft']
    self.trainArgs = config['trainArgs']
    self.genCf = config['general']
    self.trainCf = config['qagTrainer']
    if (self.genCf['ignoreWarnings'] == 'True'): self.warningIgnore()
    mode = self.trainCf['mode']
    self.maxSteps = int(self.trainArgs[f'max{mode.capitalize()}Steps'])
    self.setUpConfig()

  def announce(self, str):
    print(f'\rQAG Trainer state: {str}')
  
  def setUpConfig(self):
    # quantized LoRA (QLoRA) - uses 4-bit normal float to lighten GPU load
    self.bnbConfig = BitsAndBytesConfig(
      load_in_4bit = True,
      # we leave the model quantized in 4 bits
      bnb_4bit_quant_type = 'nf4',
      bnb_4bit_compute_dtype = torch.float16
    )
    
    self.trainingArgs = TrainingArguments(
      output_dir=self.paths['output'],
      per_device_train_batch_size = int(self.trainArgs['perDeviceTrainBatchSize']),
      gradient_accumulation_steps = int(self.trainArgs['gradientAccumulationSteps']),
      learning_rate = float(self.trainArgs['learningRate']),
      logging_steps = int(self.trainArgs['saveSteps']),
      max_steps = self.maxSteps,
      logging_dir = self.paths['log'],
      save_strategy = self.trainArgs['saveStrategy'],
      save_steps = min(int(self.trainArgs['saveSteps']), self.maxSteps),
      evaluation_strategy = self.trainArgs['evaluationStrategy'],
      eval_steps = int(self.trainArgs['evalSteps']),
      do_eval = self.trainArgs['doEval'] == 'True',
      report_to = self.trainArgs['reportTo']
    )

    self.peftConfig = LoraConfig(
      lora_alpha = int(self.peft['loraAlpha']),
      lora_dropout = float(self.peft['loraDropout']),
      r = int(self.peft['r']),
      bias = self.peft['bias'],
      task_type = self.peft['taskType']
    )
  
  def loadModel(self):
    # load our model
    self.baseModel = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path=self.paths['base'],
      quantization_config=self.bnbConfig,
      device_map='auto'
    )
    self.baseModel.config.use_cache = False

    # more info: https://github.com/huggingface/transformers/pull/24906
    self.baseModel.config.pretraining_tp = 1 

    # load our tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.paths['base'])
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # add custom tokens here
    if (self.trainCf['addCustomTokens'] == 'True'): self.addCustomTokens()
  
  def addCustomTokens(self):
    newTokens = ['<hl>']
    vocabulary = self.tokenizer.get_vocab().keys()
    for token in newTokens:
      # check to see if new token is in the vocabulary or not
      if token not in vocabulary:
        self.tokenizer.add_tokens(token)

    self.baseModel.resize_token_embeddings(len(self.tokenizer))

  def train(self, dataFormatter: DataFormatter):
    collator = None
    if (self.trainCf['optimizeCompletion'] == 'True'):
      response_template = " ### Answer:"
      collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
    
    # use the SFTTrainer from HuggingFace's trl
    trainer = SFTTrainer(
        model=self.baseModel,
        train_dataset=dataFormatter.trainDataset,
        eval_dataset=dataFormatter.evalDataset,
        peft_config=self.peftConfig,
        formatting_func=dataFormatter.getExamples,
        max_seq_length=int(self.trainArgs['maxSeqLength']),
        tokenizer=self.tokenizer,
        args=self.trainingArgs,
        packing=self.trainCf['packing'] == 'True',
        # i can pass in my own evaluation method here
        data_collator=collator,
      )
  
    # pass in resume_from_checkpoint=True to resume from a checkpoint
    # click on wandb.ai link for training info
    trainer.train()

  # testing models we just trained
  def testInference(self, dataFormatter: DataFormatter):
    # must first loadModel()
    loraLocation = f'{self.paths["output"]}/checkpoint-{self.maxSteps}'
    self.fineTunedModel = PeftModel.from_pretrained(self.baseModel, loraLocation)

    evalPrompt = dataFormatter.getEvalSample()
    modelInput = self.tokenizer(evalPrompt, return_tensors='pt').to('cuda')

    self.fineTunedModel.eval()
    with torch.no_grad():
      tokens = self.fineTunedModel.generate(**modelInput, max_new_tokens=100)[0]
      print(self.tokenizer.decode(tokens, skip_special_tokens=True))
  
  def testInferenceLoop(self, dataFormatter: DataFormatter):
    cmd = input('Enter to continue, anything else to quit.')
    while not cmd:
      self.testInference(dataFormatter)
      cmd = input()

  # misc f(x)s
  def warningIgnore(self):
    import warnings # i import here and hide this
    warnings.filterwarnings('ignore', category = DeprecationWarning)
    warnings.filterwarnings('ignore', category = FutureWarning)


if __name__ == '__main__':
  trainer = QAGTrainer()
  trainer.loadModel()
  df = DataFormatter()
  trainer.train(df)
  trainer.testInference(df)
