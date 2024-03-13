# import libraries we need
from configBase import ConfigBase
import time, os

class TimeLogger(ConfigBase):
  mode = ''
  model = ''

  def configure(self):
    self.mode = self.cp['general']['mode']
    self.logDir = self.paths['log']
    self.logFile = f'{self.logDir}/executionTime.txt'
    self.startTime = None
    self.stopTime = None
    self.timing = False
    # first-time setup
    if not os.path.isfile(self.logFile):
      os.makedirs(self.logDir, exist_ok=True)
      f = open(self.logFile, "w")
      f.write('model,mode,seconds\n')
    
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
    if log:
      f = open(self.logFile, "a")
      f.write(f'{self.model},{self.mode},{round(elapsedSeconds, 3)}\n')
      f.close()
    return elapsedSeconds
    
if __name__ == '__main__':
  pass
