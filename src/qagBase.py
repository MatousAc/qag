import configparser

class QAGBase:
  def __init__(self, configFilePath = 'qag.ini'):
    self.cp = configparser.ConfigParser()
    self.cp.read(configFilePath)
    self.paths = self.cp['paths']
    self.genCf = self.cp['general']
    if (self.genCf['ignoreWarnings'] == 'True'): self.warningIgnore()
    self.quiet = self.genCf['quiet'] == 'True'
    self.configure()
  
  def configure():
    pass
  
  def warningIgnore(self):
    import warnings # i import here and hide this
    warnings.filterwarnings('ignore', category = DeprecationWarning)
    warnings.filterwarnings('ignore', category = FutureWarning)
