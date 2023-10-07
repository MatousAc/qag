# import libraries we need
import configparser
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments 
from peft import LoraConfig
from trl import SFTTrainer
from dataProcessor import DataProcessor

class QAGTrainer:
  def __init__(self, configFilePath = 'qag.ini'):
    config = configparser.ConfigParser()
    config.read(configFilePath)
    self.paths = config['paths']
    self.peft = config['peft']
    self.trainArgs = config['trainArgs']
    self.config = config['qagTrainer']
    if bool(self.config['ignoreWarnings']): self.warningIgnore()
    self.setUpConfig()
  

  def loadModel(self):
    # load our model
    self.base_model = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path=self.paths['base'],
      quantization_config=self.bnb_config,
      device_map='auto'
    )
    self.base_model.config.use_cache = False

    # more info: https://github.com/huggingface/transformers/pull/24906
    self.base_model.config.pretraining_tp = 1 

    # load our tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.paths['base'])
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # add custom tokens here
    if bool(self.config['addCustomTokens']): self.addCustomTokens()
  
  def addCustomTokens(self):
    new_tokens = ['<hl>']
    vocabulary = self.tokenizer.get_vocab().keys()
    for token in new_tokens:
      # check to see if new token is in the vocabulary or not
      if token not in vocabulary:
        self.tokenizer.add_tokens(token)

    self.base_model.resize_token_embeddings(len(self.tokenizer))

  def train(self, dataProcessor: DataProcessor):
    # use the SFTTrainer from HuggingFace's trl
    trainer = SFTTrainer(
        model=self.base_model,
        train_dataset=dataProcessor.train_dataset,
        eval_dataset=dataProcessor.eval_dataset,
        peft_config=self.peft_config,
        formatting_func=dataProcessor.processData,
        max_seq_length=int(self.trainArgs['maxSeqLength']),
        tokenizer=self.tokenizer,
        args=self.training_args,
        # i can pass in my own evaluation method here
      )
  
    # pass in resume_from_checkpoint=True to resume from a checkpoint
    # click on wandb.ai link for training info
    trainer.train()


  def setUpConfig(self):
    # quantized LoRA (QLoRA) - uses 4-bit normal float to lighten GPU load
    self.bnb_config = BitsAndBytesConfig(
      load_in_4bit = True,
      # we leave the model quantized in 4 bits
      bnb_4bit_quant_type = 'nf4',
      bnb_4bit_compute_dtype = torch.float16
    )
    
    mode = self.config['mode']
    max_steps = int(self.config[f'max{mode.capitalize()}Steps'])
    
    self.training_args = TrainingArguments(
      output_dir=self.paths['output'],
      per_device_train_batch_size = int(self.trainArgs['perDeviceTrainBatchSize']),
      gradient_accumulation_steps = int(self.trainArgs['gradientAccumulationSteps']),
      learning_rate = float(self.trainArgs['learningRate']),
      logging_steps = int(self.trainArgs['saveSteps']),
      max_steps = max_steps,
      logging_dir = self.paths['log'],
      save_strategy = self.trainArgs['saveStrategy'],
      save_steps = int(self.trainArgs['saveSteps']),
      evaluation_strategy = self.trainArgs['evaluationStrategy'],
      eval_steps = int(self.trainArgs['evalSteps']),
      do_eval = bool(self.trainArgs['doEval']),
      report_to = self.trainArgs['reportTo']
    )

    self.peft_config = LoraConfig(
      lora_alpha = int(self.peft['loraAlpha']),
      lora_dropout = float(self.peft['loraDropout']),
      r = int(self.peft['r']),
      bias = self.peft['bias'],
      task_type = self.peft['taskType']
    )

  def warningIgnore(self):
    import warnings # i import here and hide this
    warnings.filterwarnings('ignore', category = DeprecationWarning)
    warnings.filterwarnings('ignore', category = FutureWarning)


if __name__ == '__main__':
  qagt = QAGTrainer()
  qagt.loadModel()
  dp = DataProcessor()
  qagt.train(dp)