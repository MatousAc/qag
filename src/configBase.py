import os, sys
from configparser import ConfigParser, ExtendedInterpolation
from mt import MT

class ConfigBase:
  '''A base class for all classes in src/. Sets up common
  configuration and basic utility f(x)s.'''
  def __init__(self, configFilePath = '/src/qag.ini', dataFormatter = None):
    # get repo base path
    repo = 'qag'
    self.basePath = f'{os.getcwd().split(repo)[0]}{repo}'
    self.basePath = os.path.normpath(self.basePath)
    # general config
    self.cp = ConfigParser(interpolation=ExtendedInterpolation())
    self.cp.read(os.path.normpath(self.basePath + configFilePath))
    
    self.mainCF = self.cp['main']
    self.mode = self.mainCF['mode']
    self.type = MT[self.mainCF['type']]
    self.modelSize = self.mainCF['modelSize']
    self.sep = ' <sep> ' # used in a special way

    if (self.mainCF['ignoreWarnings'] == 'True'): self.warningIgnore()
    self.quiet = self.mainCF['quiet'] == 'True'
    self.baseType = self.mainCF['baseType']
    # get terminal size for prettified output
    try: self.vw = os.get_terminal_size().columns
    except OSError: self.vw = 100


    # configure all paths to include base path
    # and normalize them for the current OS
    self.paths = self.cp['paths']
    for path in self.paths:
      self.paths[path] = os.path.normpath(self.basePath + self.paths[path])
    self.configure()

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
    if not width: width = self.vw - len(f'{label}{value}/{maximum}') - 12;
    value = min(value, maximum) # value cannot surpass maximum
    percent = value / maximum
    sys.stdout.write('\r')
    bar = '█' * int(width * percent)
    bar = bar + '-' * int(width * (1-percent))

    sys.stdout.write(f"{label} |{bar:{width}s}| {value}/{maximum} = {int(100 * percent)}% ")
    sys.stdout.flush()
  
  def printHeader(self, str = ''):
    '''Prints a terminal-wide header with str centered between '#'.'''
    side = '#' * int(0.48 * (self.vw - len(str)))
    print(f'\n{side} {str} {side}')
  
  def printReplace(self, str):
    '''Replaces current terminal line with str.'''
    l = self.vw - len(str) - 2
    print('\r' + str + (" " * l) + '\n')
    
  def printAbove(self, str):
    '''Replaces previous terminal line with str.'''
    erase = '\x1b[1A\x1b[2K'
    l = self.vw - len(str) - 2
    print(erase + str + (" " * l) + '\n')

if __name__ == '__main__':
  ConfigBase()
  print("No news is good news.")