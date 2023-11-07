# Environment Setup
## Recreate
This directory includes several complex environments. You should be able to use the various `.yml` files to create new conda environments.
```
conda create --name <envName> --file <envName>.yml
```
There are different environments available in this folder. Most environments have a local version and a `cuda` version. The cuda version is the environment to use for the GPU server. Here is an explanation of each saved environment:
### Current envs
* [qagHfCuda.yml](archive/qagHfCuda.yml) - main environment to use for this repository and thesis. use this for training LLaMA 2 with the trainer, data formatter, data processor, and other scripts in the `src/` directory.
* [qagHf.yml](archive/qagHf.yml) - the non-cuda, local version of the environment. should allow for data processing and inference based on the final model
* 
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