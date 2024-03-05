# import libraries we need
import os, sys, torch, numpy as np, evaluate
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, Seq2SeqTrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel
from qagBase import QAGBase
from dataFormatter import DataFormatter
from dataProcessor import DataProcessor
from timeLogger import TimeLogger

class QAGTrainer(QAGBase):
  def configure(self):
    self.hyp = self.cp['hyperparameters']
    self.trainArgs = self.cp['trainArgs']
    self.modelCf = self.cp['model']
    self.genEval = self.trainArgs['generativeEval'] == 'True'

    self.configureTraining()
    self.timer = TimeLogger()

  def configureTraining(self):
    '''Configures training arguments, quantization, and LoRA config.'''
    # if we're testing, we always want to save and evaluate after reaching maxSteps
    # configure wandb naming
    os.environ["WANDB_PROJECT"] = self.trainFor
    testSteps = int(self.trainArgs['testSteps'])
    stepSize = testSteps if self.mode == 'test' else int(self.trainArgs['stepSize'])

    self.trainingArgs = Seq2SeqTrainingArguments(
      # tunable hyperparameters
      learning_rate = float(self.hyp['learningRate']),
      weight_decay = float(self.hyp['weightDecay']),
      num_train_epochs = float(self.hyp['epochs']),
      # general
      max_steps = testSteps if self.mode == 'test' else 0, # 0 does not override epoch number
      predict_with_generate = self.genEval,
      greater_is_better = self.genEval,
      # wandb setup
      # SFTTrainer auto reports to wandb if installed. put 'none' below to turn off
      report_to = 'none' if self.mode == 'test' else 'wandb',
      run_name = os.path.split(self.outputDir)[1], # name = output folder,
      # GPU settings
      per_device_train_batch_size = int(self.trainArgs['perDeviceTrainBatchSize']),
      gradient_accumulation_steps = int(self.trainArgs['gradientAccumulationSteps']),
      eval_accumulation_steps = int(self.trainArgs['evalAccumulationSteps']),
      # output settings
      output_dir = self.outputDir,
      logging_dir = self.outputDir + '/logs',
      save_strategy = self.trainArgs['saveStrategy'],
      evaluation_strategy = self.trainArgs['evalStrategy'],
      save_steps = stepSize,
      eval_steps = stepSize,
      logging_steps = stepSize,
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
    
    # parameter-efficient fine-tuning (PEFT) w/ low rank adapter (LoRA)
    # trains the difference in weights Δh on the side of feed-forward
    # layers (FFW). makes for faster, lighter training. see https://rb.gy/9gor5
    # https://huggingface.co/docs/peft/conceptual_guides/lora#common-lora-parameters-in-peft
    self.loraConfig = LoraConfig(
      lora_alpha = int(self.hyp['loraAlpha']),
      lora_dropout = float(self.hyp['loraDropout']),
      r = int(self.hyp['r']),
      bias = self.hyp['bias'],
      # causal lm means the lm only sees tokens to the left of what it's predicting
      task_type = 'CAUSAL_LM',
      # enable more lora layers?
      # target_modules = [f'{l}_proj' for l in self.hyp['loraLayers']] # fixme: uncomment soon
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
  
  def preprocessLogits(self, logits: torch.Tensor, labels: torch.Tensor) ->   torch.Tensor:
    '''The model gives us logits and labels as tensors. Labels are shaped as lists of token
    sequence references (e.g. in torch.Size([8, 146])) there are 8 references of length 146 each).
    Logits add one dimension to this as instead of the 'token_id' value they include an array 
    of non-normalized probabilities, the size of our tokenizer vocabulary. Since higher
    values represent higher probabilities, we just pick the index of the highest value
    in dim=-1 (last dimension) to get our 'most probable token_id\''''
    # see issue: https://github.com/huggingface/transformers/issues/15466
    return logits.argmax(dim = -1)

  def nlgMetrics(self, evalPred):
    '''Computes Bleu, RougeL, and Meteor'''
    # we receive a tuple of predictions and references.
    preds, labels = evalPred
    # decode prediction tokens because custom metrics expect plain text
    preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
    decodedPreds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
    # we ignore the -100 tokens, as those are the prompt
    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    decodedLabels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    print('Preds:')
    print(decodedPreds)
    print('Labels:')
    print(decodedLabels)
    
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    # takes: predictions (list of str): translations to score.
    #        references (list of list of str|list of str): references for each translation.
    result = {
      'rogueL': rouge.compute(predictions=decodedPreds, references=decodedLabels, 
                use_stemmer=True, use_aggregator=True, rouge_types=['rougeL'])['rougeL'],
      'bleu': bleu.compute(predictions=decodedPreds, references=decodedLabels)['bleu'],
      'meteor': meteor.compute(predictions=decodedPreds, references=decodedLabels)['meteor']
    }
    result = {key: value * 100 for key, value in result.items()}
    print(result)
    
    # calculate avg generation length
    prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()} # round for 4 decimal places
    print("Cleaned and upscaled result:")
    print(result)
    return result
  
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
      # compute_metrics = None,
      # pass custom eval here: default to 'computeAccuracy'
      compute_metrics = self.nlgMetrics if self.genEval else None,
      preprocess_logits_for_metrics = self.preprocessLogits if self.genEval else None,
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
      checkpointLocation = self.getLatestCheckpointPath(self.latestModelDir)
      model = PeftModel.from_pretrained(self.baseModel, checkpointLocation)
      self.tokenizer = AutoTokenizer.from_pretrained(checkpointLocation)
      model.resize_token_embeddings(len(self.tokenizer))
      self.timer.model = self.latestModelDir.split("/")[-1]
      self.timer.mode = self.mode
      print(f'Inference using {checkpointLocation}')
      
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
