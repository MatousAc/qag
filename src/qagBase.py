import os, sys
from configparser import ConfigParser, ExtendedInterpolation

class QAGBase:
  def __init__(self, configFilePath = '/src/qag.ini', dataFormatter = None):
    # get repo base path
    repo = 'qag'
    self.basePath = f'{os.getcwd().split(repo)[0]}{repo}'
    self.basePath = os.path.normpath(self.basePath)
    # general config
    self.cp = ConfigParser(interpolation=ExtendedInterpolation())
    self.cp.read(os.path.normpath(self.basePath + configFilePath))
    self.genCf = self.cp['general']
    self.mode = self.genCf['mode']
    self.trainFor = self.genCf["trainFor"]
    self.modelSize = self.genCf["modelSize"]

    if (self.genCf['ignoreWarnings'] == 'True'): self.warningIgnore()
    self.quiet = self.genCf['quiet'] == 'True'
    self.modelType = self.genCf["modelType"]
    # get terminal size for prettified output
    try: self.vw = os.get_terminal_size().columns
    except OSError: self.vw = 100
    if dataFormatter: self.dataFormatter = dataFormatter


    # configure all paths to include base path
    # and normalize them for the current OS
    self.paths = self.cp['paths']
    for path in self.paths:
      self.paths[path] = os.path.normpath(self.basePath + self.paths[path])
    # increment output folder number
    latestModelNum = self.getLatestModelNumber()
    self.outputDir = self.paths['output'] + str(latestModelNum + 1).zfill(2)
    self.latestModelDir = self.paths['output'] + str(latestModelNum).zfill(2)
    self.configure()

  def getLatestModelNumber(self, pipelineType: str = None):
    '''Returns the latest AE/QG model (defaults to self.mode) in
    the models/mode directory if one is present. Else -1'''
    if not pipelineType: pipelineType = self.trainFor
    parent = self.paths['output'][:self.paths['output'].find(self.mode)] + self.mode
    subfolders = [f.path for f in os.scandir(parent) if f.is_dir() and pipelineType in f.name]
    subfolderNumbers = [int(f[-2:]) for f in subfolders]
    return max(subfolderNumbers) if len(subfolders) else -1

  def configure(self):
    '''Configuration for derived class'''
    pass

  def warningIgnore(self):
    '''Turns off annoying warnings during training.'''
    import warnings # i import here and hide this
    warnings.filterwarnings('ignore', category = DeprecationWarning)
    warnings.filterwarnings('ignore', category = FutureWarning)
    import os
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

  # from: https://stackoverflow.com/a/63136181/14062356
  def printProgressBar(self, value: int, maximum: int = 100,
                       label: str = 'progress', width: str = None):
    '''Prints a labeled progress bar at stage value/maximum.'''
    if not width: width = self.vw - len(label) - 10;
    percent = value / maximum
    sys.stdout.write('\r')
    bar = '█' * int(width * percent)
    bar = bar + '-' * int(width * (1-percent))

    sys.stdout.write(f"{label} |{bar:{width}s}| {int(100 * percent)}% ")
    sys.stdout.flush()
  
  def printHeader(self, str):
    '''Prints a terminal-wide header with str centered between '~'.'''
    side = '~' * int(0.48 * (self.vw - len(str)))
    print(f'\n{side} {str} {side}')

if __name__ == '__main__':
  QAGBase()
  print("No news is good news.")