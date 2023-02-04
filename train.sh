#!/bin/bash

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# If no GPU available, set `--device cpu`

python train.py --train-file data/Muris_gene_rankings.txt.gz --val-file data/val.txt \
-j 1 --max-len 64 -b 8 --epochs 10 --device cuda \
--lr 0.003 --lr_scheduler_type cosine \
--output-dir checkpoint \
--tokenizer_dir gpt2tokenizer \
--print-freq 10

