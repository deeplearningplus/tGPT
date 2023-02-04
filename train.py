from transformers import (
    AutoModelForCausalLM, GPT2Config,
    get_scheduler,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling
)

import numpy as np
import datetime
import random
import math
import time
import gzip
import os

from torch.optim import AdamW
import torch.utils.data
import torch.nn as nn
import torch

import utils

def open_file(text_file):
    isGzFile = False
    if text_file.endswith('.gz'):
        f = gzip.open(text_file)
        isGzFile = True
    else:
        f = open(text_file)
    return isGzFile, f

class GeneRankingDataset(torch.utils.data.Dataset):
    def __init__(self, text_file, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 64
        self.lines = []

        isGzFile, f = open_file(text_file)
        for line in f:
            if isGzFile:
                line = line.decode()
            if line.isspace() or len(line) == 0:
                continue
            self.lines.append(line.strip())
        f.close()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        s = self.lines[i]
        return self.tokenizer(s, add_special_tokens=True, truncation=True, is_split_into_words=False,
                              max_length=self.max_length, padding="max_length", return_tensors = 'pt')

def train_one_epoch(model, optimizer, scheduler, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        
        for k, v in batch.items():batch[k] = v.to(device, non_blocking=True)
        output = model(**batch)
        loss = output.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = batch['input_ids'].shape[0]
        
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['perplexity'].update(loss.exp().item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def evaluate(model, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            
            for k, v in batch.items():batch[k] = v.to(device, non_blocking=True)
            output = model(**batch)
            loss = output.loss
            
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = batch['input_ids'].shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['perplexity'].update(loss.exp().item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Perplexity {top1.global_avg:.3f}'.format(top1=metric_logger.perplexity))
    
    return metric_logger.perplexity.global_avg


def load_data(traindir, valdir, max_length, distributed, tokenizer):
    print("Loading data")
    dataset = GeneRankingDataset(traindir, tokenizer)
    dataset_test = GeneRankingDataset(valdir, tokenizer)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler 


def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    train_dir = args.train_file
    val_dir = args.val_file

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)

    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args.max_len, args.distributed, tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=data_collator, 
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, collate_fn=data_collator,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    config = GPT2Config(vocab_size=tokenizer.vocab_size, n_embd=1024, n_layer=8, n_head=16,
                        bos_token_id=tokenizer.bos_token_id, 
                        pad_token_id=tokenizer.pad_token_id, 
                        eos_token_id=tokenizer.eos_token_id)
    model = AutoModelForCausalLM.from_config(config)
    print(config)
        
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(len(data_loader))
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=max_train_steps)


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.dont_load_optim_sche is False:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq)
        evaluate(model, data_loader_test, device=device)
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint['model'],
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
            
            model_without_ddp.save_pretrained(args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train-file', help='training set')
    parser.add_argument('--val-file', help='validation set')
    parser.add_argument('--max-len', help='max_len [128]', type=int, default=128)
    parser.add_argument('--tokenizer_dir', help='gpt2 tokenizer', type=str)
    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.003, type=float, help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
                        metavar='W', help='weight decay (default: 0.0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--dont_load_optim_sche', help='do not load the parameters of optimizer and scheduler for resume', action='store_true')
    parser.add_argument('--random_subset', help='random splitting long sequence', action='store_true')
    parser.add_argument('--bert_config', default=None, help='bert config json file')
    parser.add_argument('--single_layer_bert', help='built a single-layer bert', action='store_true')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument("--num_warmup_steps", type=int, default=10000)
    parser.add_argument("--lr_scheduler_type", type=str,
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        default="cosine", help="The scheduler type to use.")


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
