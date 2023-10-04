from lmqg import GridSearcher

local_model_location = "/home/ac/code/aqg/models/llama-hf/7b"
local_output_location = "/home/ac/code/aqg/models/output/7bTrainedWithLmqg"
model_id = 'meta-llama/Llama-2-7b-hf'

trainer = GridSearcher(
    checkpoint_dir=local_output_location,
    dataset_path='lmqg/qg_squad',
    model=local_model_location,
    epoch=15,
    epoch_partial=5,
    batch=64,
    n_max_config=5,
    gradient_accumulation_steps=[2, 4], 
    lr=[1e-04, 5e-04, 1e-03],
    label_smoothing=[0, 0.15]
)
trainer.run()