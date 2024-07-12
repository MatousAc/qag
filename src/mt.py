from enum import Enum

class MT(Enum):
  '''MT = Model Type. The different way to format data, train
  a model, and run inference. AE and QG are used side by side.'''
  AE = 'AE'
  QG = 'QG'
  E2E = 'E2E'