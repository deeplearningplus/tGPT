#!/bin/bash

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 train.py
-j 1 --max-len 64 \
--lr 0.003 --lr_scheduler_type cosine \
--output-dir checkpoint \
--tokenizer_dir gpt2tokenizer \
--print-freq 100



