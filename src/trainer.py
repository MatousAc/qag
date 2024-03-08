# import libraries we need
import os, sys, torch, evaluate, wandb, yaml, numpy as np, pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, Seq2SeqTrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel
from modelHandler import ModelHandler

class Trainer(ModelHandler):
  '''A class that handles model training, evaluation during training,
  and some minimal inference after trainings.'''
  # prep
  def startup(self):
    self.trainConfig = self.cp['train']
    self.hyp = self.cp['hyperparameters']
    self.genEval = self.trainConfig['generativeEval'] == 'True'
    self.sweeping = False

  def configureTraining(self, hp):
    '''Configures training arguments, quantization, and LoRA config.'''
    # configure wandb naming
    os.environ['WANDB_PROJECT'] = 'sweep' if self.sweeping else self.trainFor
    os.environ['WANDB_LOG_MODEL'] = 'true'
    testSteps = int(self.trainConfig['testSteps'])

    # if we're testing, we always want to save and evaluate after reaching maxSteps
    stepSize = testSteps if self.mode == 'test' else int(self.trainConfig['stepSize'])
    self.trainingArgs = Seq2SeqTrainingArguments(
      # tunable hyperparameters
      learning_rate = hp['learningRate'],
      weight_decay = hp['weightDecay'],
      num_train_epochs = float(self.hyp['epochs']),
      # general
      # maxSteps is 0 if not testing or sweeping. 0 does not override epoch number
      max_steps = testSteps if self.mode == 'test' else 200 if self.sweeping else 0,
      predict_with_generate = self.genEval,
      greater_is_better = self.genEval,
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
  def preprocessLogits(self, logits: torch.Tensor, labels: torch.Tensor) ->   torch.Tensor:
    '''The model gives us logits and labels as tensors. Labels are shaped as lists of token
    sequence references (e.g. in torch.Size([8, 146])) there are 8 references of length 146 each).
    Logits add one dimension to this as instead of the 'token_id' value they include an array 
    of non-normalized probabilities, the size of our tokenizer vocabulary. Since higher
    values represent higher probabilities, we just pick the index of the highest value
    in dim=-1 (last dimension) to get our 'most probable token_id\''''
    # see issue: https://github.com/huggingface/transformers/issues/15466
    return logits.argmax(dim = -1)

  def decodeEvalPred(self, evalPred):
    '''Uses the evalPred object provided by the STFTrainer
    to return predictions and labels for training evaluation.'''
    # we receive a tuple of predictions and references.
    preds, labels = evalPred
    # decode prediction tokens because custom metrics expect plain text
    preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
    decodedPreds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
    # we ignore the -100 tokens, as those are the prompt
    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    decodedLabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    return (decodedPreds, decodedLabels)

  def generateEvalPred(self):
    '''Uses  the last checkpoint's Peft Models to generate
    predictions, and uses evaluation data for labels.'''
    model = self.getLatestCheckpoint(self.outputDir)
    inputs, labels = self.df.getEvalInputs()
    preds = []
    for inp in inputs: preds.append(self.infer(model, inp))
    preds = [pred.split(self.df.respKey)[1].strip() for pred in preds]
    return (preds, labels)
    
  def nlgMetrics(self, evalPred):
    '''Computes Bleu, RougeL, and Meteor'''
    if self.trainConfig['useEvalPred'] == 'True': preds, labels = self.decodeEvalPred(evalPred)
    else: preds, labels = self.generateEvalPred()
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    # takes: predictions (list of str): translations to score.
    #        references (list of list of str|list of str): references for each translation.
    result = {
      'rogueL': rouge.compute(predictions=preds, references=labels, 
                use_stemmer=True, use_aggregator=True, rouge_types=['rougeL'])['rougeL'],
      'bleu': bleu.compute(predictions=preds, references=labels)['bleu'],
      'meteor': meteor.compute(predictions=preds, references=labels)['meteor']
    }
    result = {key: value * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()} # round for 4 decimal places
    return result
  
  def train(self, config = None):
    '''Sets up and conducts fine-tuning'''
    # always load model first
    baseModel = self.loadModel()
    
    # multiplex hyperparams
    hp = {
      'learningRate': float(self.hyp['learningRate']),
      'weightDecay': float(self.hyp['weightDecay']),
      'r': int(self.hyp['r']),
      'loraAlpha': int(self.hyp['loraAlpha']),
      'loraDropout': float(self.hyp['loraDropout']),
      'loraLayers': self.hyp['loraLayers'],
      'bias': self.hyp['bias'],
    }
    
    collator = None # by passing None, we use the default collator
    if (self.trainConfig['optimizeCompletion'] == 'True'):
      collator = DataCollatorForCompletionOnlyLM(
        self.df.respKey, tokenizer=self.tokenizer
      )
    
    def getTrainer(): 
      # use the SFTTrainer from HuggingFace's trl library
      return SFTTrainer(
        model = baseModel,
        # model_init = self.loadModel,
        train_dataset = self.df.trainDataset,
        eval_dataset = self.df.evalDataset,
        peft_config = self.loraConfig,
        formatting_func = self.df.getExamples,
        max_seq_length = int(self.trainConfig['maxSeqLength']),
        tokenizer = self.tokenizer,
        args = self.trainingArgs,
        packing = self.trainConfig['packing'] == 'True',
        data_collator = collator,
        compute_metrics = self.nlgMetrics if self.genEval else None,
        preprocess_logits_for_metrics = self.preprocessLogits if self.genEval else None,
        # pass in resume_from_checkpoint=True to resume from a checkpoint
      )
    
    if self.sweeping:
      with wandb.init(config=config):
        hp = wandb.config
        self.configureTraining(hp)
        trainer = getTrainer()
        trainer.train()
        # manually log the loss
        df = pd.DataFrame(trainer.state.log_history)
        # print(df)
        wandb.log({
          'loss': df['loss'][0],
          'total_flos': df['total_flos'][2],
          'train_loss': df['train_loss'][2]
        })
    else:
      self.configureTraining(hp)
      getTrainer().train()
      self.printHeader('Training Success')
      print(f'Model saved to {self.outputDir}')

  def sweep(self):
    self.sweeping = True
    config = yaml.safe_load(Path(f'sweep{self.mode.capitalize()}.yml').read_text())
    sweepId = wandb.sweep(config, project = config['project'])
    wandb.agent(sweepId, self.train, count = config['iterations'])

  # testing the models
  def infer(self, model: AutoModelForCausalLM, inferenceInput = None):
    '''Infers with the specified model'''
    if not inferenceInput: inferenceInput = self.df.getInferenceInput(self.dp)
    modelInput = self.tokenizer(inferenceInput, return_tensors='pt').to('cuda')
    model.eval()
    with torch.no_grad():
      tokens = model.generate(**modelInput, max_new_tokens=100)[0]
      prediction = self.tokenizer.decode(tokens, skip_special_tokens=True)
    return prediction
  
  def inferenceLoop(self, useBase = False):
    '''Loops inference with base of fine-tuned models'''
    model = self.loadModel()
    self.timer.model = self.paths["base"].split("/")[-1]
    self.timer.mode = 'base'
    if not useBase:
      model = self.getLatestCheckpoint(self.latestModelDir)
      print(f'Inference using {self.checkpointLocation}')
      
    self.printHeader('Testing Loop')
    print('Ctrl+C to exit')
    try:
      while True:
        self.timer.start()
        print(self.infer(model))
        self.timer.stop()
        print('~' * self.vw)
    except KeyboardInterrupt: print('\rClosing\n')
    except: raise # rethrow

  def getLatestCheckpoint(self, modelDir):
    '''Returns the latest model ready for inference. If no models are
    present in the modelDir, returns basemodel. Sets up timer.
    Assumes no change in tokenizer from base model.'''
    self.checkpointLocation = self.getLatestCheckpointPath(modelDir)
    baseModel = self.loadModel()
    if self.checkpointLocation == False: return baseModel
    model = PeftModel.from_pretrained(baseModel, self.checkpointLocation)
    model.resize_token_embeddings(len(self.tokenizer))
    self.timer.model = self.latestModelDir.split("/")[-1]
    self.timer.mode = self.mode
    return model

if __name__ == '__main__':
  trainer = Trainer()
  if len(sys.argv) == 1: cmd = '-train'
  else: cmd = sys.argv[1]
  match cmd.replace('-', '').lower():
    case 'inferbase': trainer.inferenceLoop(useBase = True)
    case 'infer': trainer.inferenceLoop()
    case 'sweep': trainer.sweep()
    case 'train' | _: trainer.train()
