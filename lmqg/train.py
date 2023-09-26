from lmqg import GridSearcher
trainer = GridSearcher(
    checkpoint_dir='../models/llama-hf/7b',
    dataset_path='lmqg/qg_squad',
    model='t5-small',
    epoch=15,
    epoch_partial=5,
    batch=64,
    n_max_config=5,
    gradient_accumulation_steps=[2, 4], 
    lr=[1e-04, 5e-04, 1e-03],
    label_smoothing=[0, 0.15]
)
trainer.run()