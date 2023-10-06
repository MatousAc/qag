# run inside tmux:
# tmux new -s aqgTrain
# run. then detach w/ Ctrl+B, d

from dataProcessor import DataProcessor
from qagTrainer import QAGTrainer


trainer = QAGTrainer()
trainer.loadModel()
dp = DataProcessor()
trainer.train(dp)