#!/bin/bash

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

python train.py --train-file trn.txt --val-file val.txt \
-j 1 --max-len 64 -b 8 --epochs 10 --device cpu \
--lr 0.003 --lr_scheduler_type cosine \
--output-dir checkpoint \
--tokenizer_dir gpt2tokenizer \
--print-freq 2

