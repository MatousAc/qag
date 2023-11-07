# LLaMA notes
## Download models
1. [get new download link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
2. clone llama repo: https://github.com/facebookresearch/llama
3. run ./download.sh
4. paste in URL and select models
## Run from CLI
excecute

```
!pip install fire
```

test inference
```
torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 128 --max_batch_size 4
```