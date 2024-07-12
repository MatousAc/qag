# Environment Setup
## Recreate
This directory includes several complex environments. You should be able to use the various `.yml` files to create new conda environments. If you don't already use conda, you need to [install conda from here](https://docs.anaconda.com/free/miniconda/).
```
conda create --name <envName> --file <envName>.yml
```
The two environments available in this folder are for the SoC GPUs and for a local (conda) python installation. The `cuda` version is for the SoC GPUs. Note that you may have to install some cuda toolkit software before the environment installs correctly.

Here is an explanation of each saved environment:
## Current envs
* [qagHfCuda.yml](archive/qagHfCuda.yml) - main environment to use for this repository and thesis. use this for training LLaMA 2 with the trainer, data formatter, data processor, and other scripts in the `src/` directory.
* [qagHf.yml](archive/qagHf.yml) - the non-cuda, local version of the environment. should allow for data processing and inference based on the final model

### Archived Envs
* [fastT5.yml](archive/fastT5.yml) - used for generating ONNX models of the Potsawee T5 QAG model. Creating the models seemed to work, but opening them and running inference never worked.
* [qagLmqg.yml](archive/qagLmqg.yml) && [qagLmqgCuda.yml](archive/qagLmqgCuda.yml) - used to test the lmqg python package for model training. This did not work with LLaMA 2.
* [qagT5.yml](archive/qagT5.yml) && [qagT5Cuda.yml](archive/qagT5Cuda.yml) - main environments used during the attempts to train T5 for QAG. never fully worked, but showed promise. also includes packages for Optimum ONNX generation which *did* work

## Start from scratch
Otherwise start a conda environment from scratch with:

```
conda create -n aqg
conda activate aqg
conda install python=3.11.5
```

Install the python packages that you need.

For optimum and t5:
```
pip install datasets evaluate fastt5 huggingface kaggle pandas numpy onnx onnxruntime optimum tokenizers torch transformers nltk
```

For LLaMA 2 training:
```
pip install datasets evaluate huggingface numpy pandas transformers tokenizers torch
```

It's up to you to figure out how to install cuda toolkit on your machine.

# Training
To fine-tune a new model set up the proper config (AE vs. QG) and then run [trainer.py](src/trainer.py). The run stats should be available in the pbe_qag team on wandb.ai under the AE or QG project.

To run AE and the QG traing back-to-back, just start with `trainFor` in [qag.ini](src/qag.ini) as `AE` and run:
```
python trainer.py; python trainer.py
```
And then once the first training has started, simply change `trainFor` to `QG`. Save the file. This will run AE training first and, when complete, will run another trainig, but this time QG as that is what [qag.ini](src/qag.ini) specifies.

# Generating questions and answers
To run the model just run [generator.py](src/generator.py).
```
python generator.py
```
You will be in an inference loop where you can enter a verse reference for generation or press enter for a random verse.

The configuration for the project is in [qag.ini](src/qag.ini). This file determines the current model source, data source, and inference prompt used. Data && logs are kept in the [data](data/) folder.

# Resources
The [resources](resources/) folder is for miscellaneous project resources.
