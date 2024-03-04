# import libraries we need
import os, sys, torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel
from qagBase import QAGBase
from dataFormatter import DataFormatter
from dataProcessor import DataProcessor
from timeLogger import TimeLogger

class QAGTrainer(QAGBase):
  def configure(self):
    self.lora = self.cp['lora']
    self.trainArgs = self.cp['trainArgs']
    self.modelCf = self.cp['model']
    
    # configure wandb naming
    os.environ["WANDB_PROJECT"] = self.trainFor
    self.runName = os.path.split(self.outputDir)[1] # run name = output folder
    
    self.configureTraining()
    self.timer = TimeLogger()

  def configureTraining(self):
    '''Configures training arguments, quantization, and LoRA config.'''
    # if we're testing, we always want to save and evaluate after reaching maxSteps
    self.trainingArgs = TrainingArguments(
      output_dir = self.outputDir,
      per_device_train_batch_size = int(self.trainArgs['perDeviceTrainBatchSize']),
      gradient_accumulation_steps = int(self.trainArgs['gradientAccumulationSteps']),
      learning_rate = float(self.trainArgs['learningRate']),
      logging_steps = int(self.trainArgs['stepSize']),
      num_train_epochs = float(self.trainArgs['epochs']),
      max_steps = 1 if self.mode == 'test' else 0, # 1 step for tests, 0 does not override epoch number
      logging_dir = self.outputDir + '/logs',
      save_strategy = self.trainArgs['saveAndEvalStrategy'],
      save_steps = int(self.trainArgs['stepSize']),
      evaluation_strategy = self.trainArgs['saveAndEvalStrategy'],
      eval_steps = int(self.trainArgs['stepSize']),
      # SFTTrainer auto reports to wandb if installed. put 'none' below to turn off
      report_to = 'none' if self.mode == 'test' else 'wandb',
      run_name = self.runName,
      eval_accumulation_steps = int(self.trainArgs['evalAccumulationSteps']),
      save_total_limit = int(self.trainArgs['saveTotalLimit']),
      load_best_model_at_end = self.trainArgs['loadBestModelAtEnd'] == 'True',
    )
    
    # quantized LoRA (QLoRA) - uses 4-bit normal float to lighten GPU load
    self.bnbConfig = BitsAndBytesConfig(
      load_in_4bit = True,
      # we leave the model quantized in 4 bits
      bnb_4bit_quant_type = 'nf4',
      bnb_4bit_compute_dtype = torch.float16
    )
    
    self.loraConfig = LoraConfig(
      lora_alpha = int(self.lora['loraAlpha']),
      lora_dropout = float(self.lora['loraDropout']),
      r = int(self.lora['r']),
      bias = self.lora['bias'],
      # causal lm means the lm only sees tokens to the left of what it's predicting
      task_type = 'CAUSAL_LM',
      # enable more lora layers
      # target_modules=["q_proj", "v_proj"] # fixme: uncomment soon
    )
  
  def loadModel(self):
    '''Loads the base model and tokenizer'''
    self.baseModel = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path=self.paths['base'],
      quantization_config=self.bnbConfig,
      device_map='auto'
    )

    # cache is incompatible with gradient checkpointing
    self.baseModel.config.use_cache = False
    # more info: https://huggingface.co/docs/transformers/en/model_doc/llama2
    # increase for slower but more accurate computation
    self.baseModel.config.pretraining_tp = 1

    # load our tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.paths['base'])
    # add custom/padding tokens
    if (self.modelCf['addCustomTokens'] == 'True'): self.addCustomTokens()
    else: self.tokenizer.pad_token = self.tokenizer.eos_token
  
  def addCustomTokens(self):
    '''Adds custom tokens to model. Don't use this option.'''
    specialTokens = {
      "pad_token":"<pad>",
      "sep_token":"<sep>",
    }
    numAddedToks = self.tokenizer.add_special_tokens(specialTokens)
    if not self.quiet: print(f'Added {numAddedToks} tokens.')
    self.baseModel.resize_token_embeddings(len(self.tokenizer))
  
  def train(self):
    '''Sets up and conducts fine-tuning'''
    collator = None # by passing None, we use the default collator
    if (self.modelCf['optimizeCompletion'] == 'True'):
      collator = DataCollatorForCompletionOnlyLM(
        self.dataFormatter.respKey, tokenizer=self.tokenizer
      )
    
    # use the SFTTrainer from HuggingFace's trl library
    trainer = SFTTrainer(
      model=self.baseModel,
      train_dataset = self.dataFormatter.trainDataset,
      eval_dataset = self.dataFormatter.evalDataset,
      peft_config = self.loraConfig,
      formatting_func = self.dataFormatter.getExamples,
      max_seq_length = int(self.trainArgs['maxSeqLength']),
      tokenizer = self.tokenizer,
      args = self.trainingArgs,
      packing = self.modelCf['packing'] == 'True',
      data_collator = collator,
      # pass custom eval here
      compute_metrics = None, # default to 'computeAccuracy'
    )
    # pass in resume_from_checkpoint=True to resume from a checkpoint
    # click on wandb.ai link for training info
    trainer.train()
    self.printHeader('Training Success')
    print(f'Model saved to {self.outputDir}')

  # testing the models
  def inference(self, model: AutoModelForCausalLM):
    '''Infers with the specified model'''
    inferenceInput = self.dataFormatter.getInferenceInput(self.dp)
    modelInput = self.tokenizer(inferenceInput, return_tensors='pt').to('cuda')
    self.timer.start()
    model.eval()
    with torch.no_grad():
      tokens = model.generate(**modelInput, max_new_tokens=100)[0]
      print(self.tokenizer.decode(tokens, skip_special_tokens=True))
    self.timer.stop()
    
    print('~' * self.vw)
  
  def inferenceLoop(self, useBase = False):
    '''Loops inference with base of fine-tuned models'''
    model = self.baseModel
    self.timer.model = self.paths["base"].split("/")[-1]
    self.timer.mode = 'base'
    if not useBase:
      loraLocation = f'{self.latestModelDir}/checkpoint-{self.maxSteps}'
      model = PeftModel.from_pretrained(self.baseModel, loraLocation)
      self.tokenizer = AutoTokenizer.from_pretrained(loraLocation)
      model.resize_token_embeddings(len(self.tokenizer))
      self.timer.model = self.latestModelDir.split("/")[-1]
      self.timer.mode = self.mode
      print(f'Inference using {self.latestModelDir}')
      
    self.printHeader('Testing Loop')
    print('Ctrl+C to exit')
    self.dp = DataProcessor()
    try:
      while True: self.inference(model)
    except KeyboardInterrupt: print('\rClosing\n')
    except: raise # rethrow

if __name__ == '__main__':
  df = DataFormatter()
  trainer = QAGTrainer(dataFormatter=df)
  # must first loadModel()
  trainer.loadModel()
  if len(sys.argv) == 1: cmd = '-train'
  else: cmd = sys.argv[1]
  match cmd.replace('-', '').lower():
    case 'inferbase': trainer.inferenceLoop(useBase = True)
    case 'infer': trainer.inferenceLoop()
    case 'train' | _: trainer.train()
