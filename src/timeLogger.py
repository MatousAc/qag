# import libraries we need
from qagBase import QAGBase
import time

class TimeLogger(QAGBase):
  def configure(self):
    self.mode = self.cp['qagTrainer']['mode']
    self.logDir = self.paths['log']
    self.start = None
    self.stop = None
    self.timing = False
    
  def play(self):
    if self.timing:
      print('Already timing')
      return
    
    self.timing = True
    self.start = time.time()
    
  def pause(self, log = True):
    if not self.timing:
      print('Not timing')
      return
    
    self.stop = time.time()
    self.timing = False
    elapsedSeconds = self.stop - self.start
    model = self.paths["base"].split("/")[-1]
    if log:
      f = open(f'{self.logDir}/baseModelExecutionTime.txt', "a")
      f.write(f'{model}, {round(elapsedSeconds, 3)}\n')
      f.close()
    return elapsedSeconds
    
if __name__ == '__main__':
  pass
