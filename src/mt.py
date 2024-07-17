from enum import Enum

class MT(Enum):
  '''MT = Model Type. The different way to format data, train
  a model, and run inference. AE, QG, & QA are used side by side.'''
  AE = 'AE'
  QG = 'QG'
  QA = 'QA'
  E2E = 'E2E'