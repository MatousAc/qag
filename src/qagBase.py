from configparser import ConfigParser, ExtendedInterpolation

class QAGBase:
  def __init__(self, configFilePath = 'qag.ini', dataFormatter = None):
    self.cp = ConfigParser(interpolation=ExtendedInterpolation())
    self.cp.read(configFilePath)
    self.paths = self.cp['paths']
    self.genCf = self.cp['general']
    if (self.genCf['ignoreWarnings'] == 'True'): self.warningIgnore()
    self.quiet = self.genCf['quiet'] == 'True'
    self.trainFor = self.genCf["trainFor"]
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
