# import libraries we need
from qagBase import QAGBase
import time, os

class TimeLogger(QAGBase):
  def configure(self):
    self.mode = self.cp['qagTrainer']['mode']
    self.logDir = self.paths['log']
    self.logFile = f'{self.logDir}/baseModelExecutionTime.txt'
    self.startTime = None
    self.stopTime = None
    self.timing = False
    # first-time setup
    if not os.path.isfile(self.logFile):
      os.makedirs(self.logDir, exist_ok=True)
      f = open(self.logFile, "w")
      f.write('model, seconds\n')
    
  def start(self):
    if self.timing:
      print('Already timing')
      return
    
    self.timing = True
    self.startTime = time.time()
    
  def stop(self, log = True):
    if not self.timing:
      print('Not timing')
      return
    
    self.stopTime = time.time()
    self.timing = False
    elapsedSeconds = self.stopTime - self.startTime
    model = self.paths["base"].split("/")[-1]
    if log:
      f = open(self.logFile, "a")
      f.write(f'{model}, {round(elapsedSeconds, 3)}\n')
      f.close()
    return elapsedSeconds
    
if __name__ == '__main__':
  pass
