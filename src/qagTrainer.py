# import libraries we need
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments 
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from peft import PeftModel
import evaluate
import numpy as np
from qagBase import QAGBase
from dataFormatter import DataFormatter

class QAGTrainer(QAGBase):
  def configure(self):
    self.peft = self.cp['peft']
    self.trainArgs = self.cp['trainArgs']
    self.trainCf = self.cp['qagTrainer']
    mode = self.trainCf['mode']
    self.maxSteps = int(self.trainArgs[f'max{mode.capitalize()}Steps'])
    self.configureTraining()
    
    fxStr = self.trainCf['metricFx']
    if fxStr == 'computeAccuracy': self.metricFx = None
    else: self.metricFx = evaluate.load(fxStr)

  def computeMetric(self, p): # currently broken
    # predictions = ["hello there general kenobi", "foo bar foobar"]
    # references = [["hello there general kenobi", "hello there !"],["foo bar foobar"]]
    # bleu = evaluate.load("bleu")
    # results = bleu.compute(predictions=predictions, references=references)
    # print(results)
    
    # do we have to decode our predictions?
    decoded_preds = self.detokenize(p.predictions)
    
    # replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    decoded_labels = self.detokenize(labels)
    
    # why can't I see anything I print out? I can't even log from here
    print(f'Evaluating with {self.trainCf["metricFx"].upper()}')
    print('Predictions:')
    print(p.predictions)
    print('References:')
    print(p.references)
    
    for ref, pred in zip(decoded_preds, decoded_labels):
      self.metricFx.add(references=ref, predictions=pred)
    metrics = self.metricFx.compute()
    # metrics = self.metricFx.compute(predictions=decoded_preds, references=decoded_labels,
                                    # tokenizer=self.tokenizer)
    print(metrics)
    return metrics

  def configureTraining(self):
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
    collator = None # by passing None, we use the default collator
    if (self.trainCf['optimizeCompletion'] == 'True'):
      collator = DataCollatorForCompletionOnlyLM(
        dataFormatter.respTemple, tokenizer=self.tokenizer
      )
    
    # use the SFTTrainer from HuggingFace's trl library
    trainer = SFTTrainer(
        model=self.baseModel,
        train_dataset = dataFormatter.trainDataset,
        eval_dataset = dataFormatter.evalDataset,
        peft_config = self.peftConfig,
        formatting_func = dataFormatter.getExamples,
        max_seq_length = int(self.trainArgs['maxSeqLength']),
        tokenizer = self.tokenizer,
        args = self.trainingArgs,
        packing = self.trainCf['packing'] == 'True',
        data_collator = collator,
        # pass custom eval here
        compute_metrics = None if self.metricFx == None else self.computeMetric
      )
  
    # pass in resume_from_checkpoint=True to resume from a checkpoint
    # click on wandb.ai link for training info
    trainer.train()

  def detokenize(self, tokens):
    return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
  # testing models we just trained
  def testInference(self, dataFormatter: DataFormatter):
    # must first loadModel()
    print('##### Start Inference Test #####')
    loraLocation = f'{self.paths["output"]}/checkpoint-{self.maxSteps}'
    self.fineTunedModel = PeftModel.from_pretrained(self.baseModel, loraLocation)

    evalPrompt = dataFormatter.getEvalSample()
    modelInput = self.tokenizer(evalPrompt, return_tensors='pt').to('cuda')

    self.fineTunedModel.eval()
    with torch.no_grad():
      tokens = self.fineTunedModel.generate(**modelInput, max_new_tokens=100)[0]
      print(self.detokenize(tokens))
    
    print('##### End Inference Test #####')
  
  def testInferenceLoop(self, dataFormatter: DataFormatter):
    cmd = input('Enter to continue, anything else to quit.')
    while not cmd:
      self.testInference(dataFormatter)
      cmd = input()


if __name__ == '__main__':
  trainer = QAGTrainer()
  trainer.loadModel()
  df = DataFormatter()
  trainer.train(df)
  trainer.testInference(df)
