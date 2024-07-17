# import libraries we need
import os, sys, torch, wandb, yaml, numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, Seq2SeqTrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel
from modelHandler import ModelHandler
from mt import MT

class Trainer(ModelHandler):
  '''A class that handles model training, evaluation during training,
  and some minimal inference after trainings.'''
  # prep
  def startup(self):
    self.trainConfig = self.cp['train']
    self.hyp = self.cp['hyperparameters']
    self.sweeping = False

  def configureTraining(self, hp):
    '''Configures training arguments, quantization, and LoRA config.'''
    # configure wandb naming
    os.environ['WANDB_PROJECT'] = 'sweep' if self.sweeping else self.type.value
    os.environ['WANDB_LOG_MODEL'] = 'true'
    testSteps = int(self.trainConfig['testSteps'])

    # if we're testing or sweeping, we always want to save and evaluate after reaching maxSteps
    stepSize = testSteps if self.mode == 'test' else self.maxSteps if self.sweeping else int(self.trainConfig['stepSize'])
    self.trainingArgs = Seq2SeqTrainingArguments(
      # tunable hyperparameters
      learning_rate = hp['learningRate'],
      weight_decay = float(self.hyp['weightDecay']),
      num_train_epochs = float(self.hyp['epochs']),
      # general
      # maxSteps is 0 if not testing or sweeping. 0 does not override epoch number
      max_steps = testSteps if self.mode == 'test' else self.maxSteps if self.sweeping else 0,
      # wandb setup
      # SFTTrainer auto reports to wandb if installed. put 'none' below to turn off
      report_to = 'none' if self.sweeping or self.mode == 'test' else 'wandb',
      run_name = os.path.split(self.outputDir)[1], # name = output folder,
      # GPU settings
      per_device_train_batch_size = int(self.trainConfig['perDeviceTrainBatchSize']),
      gradient_accumulation_steps = int(self.trainConfig['gradientAccumulationSteps']),
      eval_accumulation_steps = int(self.trainConfig['evalAccumulationSteps']),
      # output settings
      output_dir = self.outputDir,
      logging_dir = self.outputDir + '/logs',
      save_strategy = 'no' if self.sweeping else 'steps' if self.mode == 'tests' else self.trainConfig['saveStrategy'],
      evaluation_strategy = self.trainConfig['evalStrategy'],
      save_steps = stepSize,
      eval_steps = stepSize,
      logging_steps = stepSize,
      save_total_limit = int(self.trainConfig['saveTotalLimit']),
      load_best_model_at_end = self.trainConfig['loadBestModelAtEnd'] == 'True',
    )
    
    # parameter-efficient fine-tuning (PEFT) w/ low rank adapter (LoRA)
    # trains the difference in weights Δh on the side of feed-forward
    # layers (FFW). makes for faster, lighter training. see https://rb.gy/9gor5
    # https://huggingface.co/docs/peft/conceptual_guides/lora#common-lora-parameters-in-peft
    self.loraConfig = LoraConfig(
      lora_alpha = int(hp['loraAlpha']),
      lora_dropout = float(hp['loraDropout']),
      r = int(hp['r']),
      # causal lm means the lm only sees tokens to the left of what it's predicting
      task_type = 'CAUSAL_LM',
      # enable more lora layers
      target_modules = [f'{l}_proj' for l in hp['loraLayers']],
      bias = hp['bias'],
    )
  
  def loadModel(self):
    '''Loads the base model and tokenizer'''
    # quantized LoRA (QLoRA) - uses 4-bit normal float to lighten GPU load
    self.bnbConfig = BitsAndBytesConfig(
      load_in_4bit = True,
      # we leave the model quantized in 4 bits
      bnb_4bit_quant_type = 'nf4',
      bnb_4bit_compute_dtype = torch.float16
    )
    
    baseModel = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path=self.paths['base'],
      quantization_config=self.bnbConfig,
      device_map='auto'
    )

    # cache is incompatible with gradient checkpointing
    baseModel.config.use_cache = False
    # more info: https://huggingface.co/docs/transformers/en/model_doc/llama2
    # increase for slower but more accurate computation
    baseModel.config.pretraining_tp = 1

    # load our tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.paths['base'])
    # add custom/padding tokens
    if (self.trainConfig['addCustomTokens'] == 'True'): self.addCustomTokens(baseModel)
    else: self.tokenizer.pad_token = self.tokenizer.eos_token
    return baseModel

  def addCustomTokens(self, model: AutoModelForCausalLM):
    '''Adds custom tokens to model. Don't use this option.'''
    specialTokens = {
      "pad_token":"<pad>",
      "sep_token":"<sep>",
    }
    numAddedToks = self.tokenizer.add_special_tokens(specialTokens)
    if not self.quiet: print(f'Added {numAddedToks} tokens.')
    return model.resize_token_embeddings(len(self.tokenizer))
  
  # train
  def train(self, config = None):
    '''Sets up and conducts fine-tuning'''
    # always load model first
    baseModel = self.loadModel()
    
    # default source of hyperparameters
    hp = {
      'learningRate': float(self.hyp['learningRate']),
      'r': int(self.hyp['r']),
      'loraAlpha': int(self.hyp['loraAlpha']),
      'loraDropout': float(self.hyp['loraDropout']),
      'loraLayers': self.hyp['loraLayers'],
      'bias': self.hyp['bias'],
      'quality': int(self.cp['data']['qualityThreshold']),
    }
    
    collator = None # by passing None, we use the default collator
    if (self.trainConfig['optimizeCompletion'] == 'True'):
      collator = DataCollatorForCompletionOnlyLM(
        self.df.respTemple, tokenizer=self.tokenizer
      )
    
    def getTrainer(): 
      # use the SFTTrainer from HuggingFace's trl library
      return SFTTrainer(
        model = baseModel,
        train_dataset = self.df.trainDataset,
        eval_dataset = self.df.evalDataset,
        peft_config = self.loraConfig,
        formatting_func = self.df.getExamples,
        max_seq_length = int(self.trainConfig['maxSeqLength']),
        tokenizer = self.tokenizer,
        args = self.trainingArgs,
        packing = self.trainConfig['packing'] == 'True',
        data_collator = collator,
        # pass in resume_from_checkpoint=True to resume from a checkpoint
      )
    
    if self.sweeping:
      with wandb.init(config=config):
        # fine-tuning the best data quality
        hp.update(wandb.config) # override vals from qag.ini
        self.df.load(threshold = hp['quality'], shuffle = True)
        self.configureTraining(hp)
        trainer = getTrainer()
        trainer.train()
        # manually log the loss
        log = {}
        for dic in trainer.state.log_history: log.update(dic)
        wandb.log({
          'eval_loss': log['eval_loss'],
          'loss': log['loss'],
          'total_flos': log['total_flos'],
          'train_loss': log['train_loss'],
          'train_runtime': log['train_runtime'],
          'train_samples_per_second': log['train_samples_per_second'],
          'step': log['step'],
        })
    else:
      self.configureTraining(hp)
      getTrainer().train()
      self.printHeader('Training Success')
      print(f'Model saved to {self.outputDir}')

  def sweep(self):
    self.sweeping = True
    self.maxSteps = 200
    config = yaml.safe_load(Path(f'sweep.yml').read_text())
    sweepId = wandb.sweep(config, project = config['project'])
    wandb.agent(sweepId, self.train, count = config['iterations'])

  # rudinmentary inference
  def inferenceLoop(self):
    '''Loops inference with fine-tuned models'''
    self.model = self.loadModel()
    self.loadLora()
    self.printHeader('Testing Loop')
    print('Ctrl+C to exit')
    try:
      while True:
        print(self.infer(self.df.getInferenceInput(self.dp), reduce = False))
        self.printHeader('Example')
    except KeyboardInterrupt: self.printReplace('Closing')
    except: raise # rethrow

if __name__ == '__main__':
  trainer = Trainer()
  if len(sys.argv) == 1: cmd = '-train'
  else: cmd = sys.argv[1]
  match cmd.replace('-', '').lower():
    case 'infer': trainer.inferenceLoop()
    case 'sweep': trainer.sweep()
    case 'train' | _: trainer.train()
