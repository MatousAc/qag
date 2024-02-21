import os, sys
from configparser import ConfigParser, ExtendedInterpolation

class QAGBase:
  def __init__(self, configFilePath = '/src/qag.ini', dataFormatter = None):
    # get repo base path
    repo = 'qag'
    basePath = f'{os.getcwd().split(repo)[0]}{repo}'
    basePath = os.path.normpath(basePath)
    # get config
    self.cp = ConfigParser(interpolation=ExtendedInterpolation())
    self.cp.read(os.path.normpath(basePath + configFilePath))

    # configure all paths to include base path
    # and normalize them for the current OS
    self.paths = self.cp['paths']
    for path in self.paths:
      self.paths[path] = os.path.normpath(basePath + self.paths[path])
    
    self.genCf = self.cp['general']
    if (self.genCf['ignoreWarnings'] == 'True'): self.warningIgnore()
    self.quiet = self.genCf['quiet'] == 'True'
    self.trainFor = self.genCf["trainFor"]
    self.modelType = self.genCf["modelType"]
    # get terminal size for prettified output
    try: self.vw = os.get_terminal_size().columns
    except OSError: self.vw = 100
    if dataFormatter: self.dataFormatter = dataFormatter
    self.configure()

  def configure(self):
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
  
  def printHeader(self, str):
    side = '~' * int(0.48 * (self.vw - len(str)))
    print(f'\n{side} {str} {side}')
  
if __name__ == '__main__':
  QAGBase()
  print("No news is good news.")