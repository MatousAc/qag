# import libraries we need
import os, sys, torch, numpy as np, evaluate
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel
from qagBase import QAGBase
from dataFormatter import DataFormatter
from dataProcessor import DataProcessor
from timeLogger import TimeLogger

class QAGTrainer(QAGBase):
  def configure(self):
    self.peft = self.cp['peft']
    self.trainArgs = self.cp['trainArgs']
    self.trainCf = self.cp['qagTrainer']
    self.maxSteps = int(self.trainArgs[f'max{self.mode.capitalize()}Steps'])
    
    # configure wandb naming
    os.environ["WANDB_PROJECT"] = self.trainFor
    self.runName = os.path.split(self.outputDir)[1] # run name = output folder
    
    self.configureTraining()
    self.timer = TimeLogger(self.mode)

    # assign metric functions
    fxStr = self.trainCf['metricFx']
    if fxStr == 'computeAccuracy': self.metricFx = None
    else: self.metricFx = evaluate.load(fxStr)

  def preprocessLogits(self, logits: torch.Tensor, labels: torch.Tensor) ->   torch.Tensor:
    # at this point the model gives us logits and labels as tensors. labels seem to be shaped as
    # lists of token sequence references (e.g. in torch.Size([8, 146])) there are 8 references 
    # of length 146 each). note that the important labels are padded by a lot of -100 tokens which
    # we should filter out. logits add one dimension to this as instead of the 'token_id' value
    # they include an array of non-normalized probabilities, one for every token. this leaves us
    # with a huge z-dimension (e.g. torch.Size([8, 146, 32000])) that is basically the size of
    # our tokenizer vocabulary. the higher values represent higher probabilities.
    # thus, we can just pick the index of the highest value in dim=-1 (last dimension) to 
    # get our 'most probable token_id'
    # see issue: https://github.com/huggingface/transformers/issues/15466
    # unfortunately, just picking the highest probability in each tensor is not sufficient
    # to produce grammatically valid sentences

    # FIXME: use top-k or top-p methods of selecting tokens
    # try seeing what the library does by default
    # also remove this debugging stuff below evetually
    # print('Looking at Logits\nAll Logits in one:')
    # print(f'logits.shape: {logits.shape}')
    # print(logits)
    # print('Looking at Lables\nAll Labels in one:')
    # print(f'labels.shape: {labels.shape}')
    # print(labels)
    
    # logits = logits.argmax(dim=-1)
    # print('Looking at Logits AGAIN\nAll Logits in one:')
    # print(f'logits.shape: {logits.shape}')
    # print(logits)
      
    # print('Done w/ Preprocessing Logits')
    return logits.argmax(dim = -1)
    
  def injectCustomEvaluation(self, evalPred):
    # this function should just output the performance of the latests checkpoint
    # curretly, it does no such thing
    print(f'Evaluating with Gooogle Bleu')
    
    inputs = self.dataFormatter.getEvalInputs(20)
    print(inputs)
    prompts = []
    reference = []
    for put in inputs:
      splt = put.split(self.dataFormatter.respTemple)
      prompts.append(splt[0] + self.dataFormatter.respTemple)
      reference.append([splt[1]])
    print('Prompts')
    print(prompts)
    print('Reference')
    print(reference)
    
    
    
    # takes: predictions (list of str): list of translations to score.
    #        references (list of list of str): list of lists of references for each translation.
    metrics = self.metricFx.compute(predictions=prompts, references=reference)
    print(metrics)
    return metrics

  def computeMetric(self, evalPred):
    # we receive a tuple of predictions and references.
    # these should be triply-nested lists of ints
    print(f'Evaluating with Google Bleu')
    preds, labels = evalPred
    
    # we have to decode our predictions for most custom metrics
    print('Predictions:')
    print(preds)
    preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
    decodedPreds = [self.detokenize(ids) for ids in preds]
    print(decodedPreds)
    
    print('References:')
    print(labels)
    # replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    # every reference is a list of references by default
    decodedLabels = [[self.detokenize(ids)] for ids in labels]

    print(decodedLabels)
    
    # takes: predictions (list of str): list of translations to score.
    #        references (list of list of str): list of lists of references for each translation.
    metrics = self.metricFx.compute(predictions=decodedPreds, references=decodedLabels)
    print(metrics)
    return metrics

  def configureTraining(self):
    '''Configures training arguments, quantization, and LoRA config.'''
    # if we're testing, we always want to save and evaluate after reaching maxSteps
    saveAndEvalSteps = min(int(self.trainArgs['saveAndEvalSteps']), self.maxSteps)
    self.trainingArgs = TrainingArguments(
      output_dir=self.outputDir,
      per_device_train_batch_size = int(self.trainArgs['perDeviceTrainBatchSize']),
      gradient_accumulation_steps = int(self.trainArgs['gradientAccumulationSteps']),
      learning_rate = float(self.trainArgs['learningRate']),
      logging_steps = saveAndEvalSteps,
      max_steps = self.maxSteps,
      logging_dir = self.outputDir + '/logs',
      save_strategy = self.trainArgs['saveAndEvalStrategy'],
      save_steps = saveAndEvalSteps,
      evaluation_strategy = self.trainArgs['saveAndEvalStrategy'],
      eval_steps = saveAndEvalSteps,
      # SFTTrainer auto reports to wandb if installed. put 'none' below to turn off
      report_to = 'none' if self.mode == 'test' else 'wandb',
      run_name=self.runName,
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
    
    self.peftConfig = LoraConfig(
      lora_alpha = int(self.peft['loraAlpha']),
      lora_dropout = float(self.peft['loraDropout']),
      r = int(self.peft['r']),
      bias = self.peft['bias'],
      # causal lm means the lm only sees tokens to the left of what it's predicting
      task_type = 'CAUSAL_LM'
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
    specialTokens = [
      {'highlight_token': '<hl>'},
      {"pad_token":"<pad>"},
      {"sep_token":"<sep>"},
    ]
    numAddedToks = self.tokenizer.add_special_tokens(specialTokens)
    if not self.quiet: print(f'Added {numAddedToks} tokens.')
    self.baseModel.resize_token_embeddings(len(self.tokenizer))
    
  def train(self):
    collator = None # by passing None, we use the default collator
    if (self.trainCf['optimizeCompletion'] == 'True'):
      collator = DataCollatorForCompletionOnlyLM(
        self.dataFormatter.respKey, tokenizer=self.tokenizer
      )
    
    # use the SFTTrainer from HuggingFace's trl library
    trainer = SFTTrainer(
        model=self.baseModel,
        train_dataset = self.dataFormatter.trainDataset,
        eval_dataset = self.dataFormatter.evalDataset,
        peft_config = self.peftConfig,
        formatting_func = self.dataFormatter.getExamples,
        max_seq_length = int(self.trainArgs['maxSeqLength']),
        tokenizer = self.tokenizer,
        args = self.trainingArgs,
        packing = self.trainCf['packing'] == 'True',
        data_collator = collator,
        # pass custom eval here
        compute_metrics = None if self.metricFx == None else self.computeMetric,
        # reduce logits into token_ids for custom metrics
        preprocess_logits_for_metrics = None if self.metricFx == None else self.preprocessLogits
      )
    # pass in resume_from_checkpoint=True to resume from a checkpoint
    # click on wandb.ai link for training info
    trainer.train()
    self.printHeader('Training Success')
    print(f'Model saved to {self.outputDir}')

  def detokenize(self, tokens):
    with torch.no_grad():
      return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
  # testing the models
  def inference(self, model: AutoModelForCausalLM):
    inferenceInput = self.dataFormatter.getInferenceInput(self.dp)
    modelInput = self.tokenizer(inferenceInput, return_tensors='pt').to('cuda')
    # print('Model input')
    # print(modelInput)

    self.timer.start()
    model.eval()
    with torch.no_grad():
      tokens = model.generate(**modelInput, max_new_tokens=100)[0]
      print(self.detokenize(tokens))
    self.timer.stop()
    
    print('~' * self.vw)
  
  def inferenceLoop(self, useBase = False):
    model = self.baseModel
    self.timer.model = self.paths["base"].split("/")[-1]
    if not useBase:
      loraLocation = f'{self.latestModelDir}/checkpoint-{self.maxSteps}'
      model = PeftModel.from_pretrained(self.baseModel, loraLocation)
      self.timer.model = self.latestModelDir.split("/")[-1]
      self.timer.mode = 'base'
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
  match cmd:
    case '-inferBase': trainer.inferenceLoop(useBase = True)
    case '-infer': trainer.inferenceLoop()
    case '-train' | _: trainer.train()
