# NVIDIA Notes

to see which GPUs are being used:
```
nvidia-smi
```

install PyTorch with cuda:
see [here](https://pytorch.org/get-started/locally/) or:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

to choose which GPU to use:
```
export CUDA_VISIBLE_DEVICES=1
```