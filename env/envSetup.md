# Environment Setup
## Recreate
This directory includes several complex environments. You should be able to use the various `.yml` files to create new conda environments.
```
conda create --name <envName> --file <envName>.yml
```
There are different environments available in this folder. Here is an explanation of each:
### Archived Envs
* [fastT5.yml](fastT5.yml) - used for generating ONNX models of the Potsawee T5 QAG model. Creating the models seemed to work, but opening them and running inference never worked.
* [fastT5.yml](fastT5.yml) - 

## Start from scratch
Otherwise start a conda environment from scratch with:

```
conda create -n aqg
conda activate aqg
conda install python=3.10.10
```

Then install python packages. For 
```
pip install datasets evaluate fastt5 huggingface kaggle pandas numpy onnx onnxruntime optimum tokenizers torch transformers nltk
```