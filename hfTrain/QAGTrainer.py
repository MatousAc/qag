# import libraries

# try to run this without using huggingface_hub
# from huggingface_hub import notebook_login
# notebook_login()

import configparser
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments 
from peft import LoraConfig
from trl import SFTTrainer
from dataProcessor import DataProcessor

class QAGTrainer:
  def __init__(self, configFilePath = 'QAG.ini'):
    config = configparser.ConfigParser()
    config.read(configFilePath)
    self.paths = config['paths']
    self.peft = config['peft']
    self.trCfg = config['train']
    self.setUpConfig()

  def loadModel(self):
    # load our model
    self.base_model = AutoModelForCausalLM.from_pretrained(
      base_model_dir=self.paths['base'],
      quantization_config=self.bnb_config,
      device_map='auto',
      # research what this is and why i need/don't need it
      # trust_remote_code=True,
      # use_auth_token=True
    )
    self.base_model.config.use_cache = False

    # more info: https://github.com/huggingface/transformers/pull/24906
    self.base_model.config.pretraining_tp = 1 

    # load our tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.paths['base'])
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # add custom tokens here
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
        max_seq_length=int(self.trCfg['maxSeqLength']),
        tokenizer=self.tokenizer,
        args=self.training_args,
      )
  
    # pass in resume_from_checkpoint=True to resume from a checkpoint
    # when we train, we can see our progress and system info on wandb.ai
    trainer.train()


  def setUpConfig(self):
    # cofigures ???
    self.bnb_config = BitsAndBytesConfig(
      load_in_4bit = True,
      # we leave the model quantized in 4 bits
      bnb_4bit_quant_type = 'nf4',
      bnb_4bit_compute_dtype = torch.float16
    )
    
    self.training_args = TrainingArguments(
      output_dir=self.paths['output'],
      per_device_train_batch_size = int(self.trCfg['perDeviceTrainBatchSize']),
      gradient_accumulation_steps = int(self.trCfg['gradientAccumulationSteps']),
      learning_rate = float(self.trCfg['learningRate']),
      logging_steps = int(self.trCfg['loggingSteps']),
      max_steps = int(self.trCfg['maxSteps']),
      logging_dir = self.paths['log'],
      save_strategy = self.trCfg['saveStrategy'],
      save_steps = int(self.trCfg['saveSteps']),
      evaluation_strategy = self.trCfg['evaluationStrategy'],
      eval_steps = int(self.trCfg['evalSteps']),
      do_eval = bool(self.trCfg['doEval'])
    )

    self.peft_config = LoraConfig(
      lora_alpha = int(self.peft['loraAlpha']),
      lora_dropout = float(self.peft['loraDropout']),
      r = int(self.peft['r']),
      bias = self.peft['bias'],
      task_type = self.peft['taskType']
    )
 
if __name__ == '__main__':
  qagt = QAGTrainer()
  qagt.loadModel()
  dp = DataProcessor()
  qagt.train(dp)