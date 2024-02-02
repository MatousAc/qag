# run inside tmux:
# tmux new -s aqgTrain
# run. then detach w/ Ctrl+B, d

from dataFormatter import DataFormatter
from qagTrainer import QAGTrainer


trainer = QAGTrainer()
trainer.loadModel()
df = DataFormatter()
trainer.train(df)
trainer.inferenceLoop(df)
