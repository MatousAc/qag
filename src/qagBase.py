import os
from configparser import ConfigParser, ExtendedInterpolation
import sys

class QAGBase:
  def __init__(self, configFilePath = 'qag.ini', dataFormatter = None):
    self.cp = ConfigParser(interpolation=ExtendedInterpolation())
    self.cp.read(configFilePath)
    self.paths = self.cp['paths']
    self.genCf = self.cp['general']
    if (self.genCf['ignoreWarnings'] == 'True'): self.warningIgnore()
    self.quiet = self.genCf['quiet'] == 'True'
    self.trainFor = self.genCf["trainFor"]
    self.sep = self.cp['dataFormatter']['sepTok']
    self.modelType = self.genCf["modelType"]
    self.vw = os.get_terminal_size().lines
    if dataFormatter: self.dataFormatter = dataFormatter
    self.configure()
  
  def configure():
    pass
  
  def warningIgnore(self):
    import warnings # i import here and hide this
    warnings.filterwarnings('ignore', category = DeprecationWarning)
    warnings.filterwarnings('ignore', category = FutureWarning)
    import os
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

  # from: https://stackoverflow.com/a/63136181/14062356
  def printProgressBar(self, value, maximum = 100, label = 'progress', width = 50):
      j = value / maximum
      sys.stdout.write('\r')
      bar = '█' * int(width * j)
      bar = bar + '-' * int(width * (1-j))

      sys.stdout.write(f"{label.ljust(10)} | [{bar:{width}s}] {int(100 * j)}% ")
      sys.stdout.flush()