from functools import partial
import plotly.express as px
import argparse
import torch.nn as nn
import torch
from .imagenet import get_imagenet
from .coco import get_coco

imshow = partial(px.imshow, template='plotly_dark')

def t2i(tensor):
    return tensor.permute(1,2,0)

def add_model_args(params,parent_parser):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,  default='')
    parser.add_argument('--mode', default='pretrain')
    known_args, _ = parser.parse_known_args(params)
    parser = parent_parser.add_argument_group("ViT")
    parser.add_argument('--decoder_embed_dim', type=int, default=96)
    if known_args.mode=='pretrain':
        parser.add_argument('--mask_ratio', type=float, default=0.25)
        parser.add_argument('--decoder_depth', type=int, default=1)
        parser.add_argument('--decoder_num_heads', type=int, default=3)
    if 'method1' in known_args.model_name:
        parser = parent_parser.add_argument_group("MAE Method1")
        parser.add_argument('--levels', type=int, default=4)
        parser.add_argument('--min_keep_ratio', type=float, default=0.5)
    elif 'method2' in known_args.model_name:
        parser = parent_parser.add_argument_group("MAE Method2")
        parser.add_argument('--levels', type=int, default=10)
        parser.add_argument('--len_keep_level', type=int, default=18)
        parser.add_argument('--fold_size', type=int, default=4)
        parser.add_argument('--folds', type=int, default=4)
        parser.add_argument('--scale', type=int, default=12)
    elif 'method3' in known_args.model_name:
        parser.add_argument('--levels', type=int, default=4)
        parser = parent_parser.add_argument_group("MAE Method3")
        parser.add_argument('--resize_size', type=int, default=32)
        parser.add_argument('--min_keep_ratio', type=float, default=0.15)
    if known_args.model_name=='yolos-method3':
        parser = parent_parser.add_argument_group("YOLOS")  
        parser.add_argument("--dataset", type=str, default="COCO")     
        parser.add_argument("--no_person", type=bool, default=False)          
        parser.add_argument("--patch_size", type=int, default=16)
    else:
        parser.add_argument("--dataset", type=str, default="ImageNet")     
    parser.add_argument('--version', type=str,default='')
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--qkv_bias', type=bool, default=True)
    parser.add_argument('--norm_layer', default=partial(nn.LayerNorm, eps=1e-6))
    return parent_parser

def get_dataloader(args):
    if args.dataset=='ImageNet':
        train_dataset,validation_dataset=get_imagenet(args)
        collate_fn=None
    elif args.dataset=='COCO':
        train_dataset,validation_dataset=get_coco(args)
        from util.coco import collate_fn

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True)
    args.steps_per_gpu=(len(train_loader)//args.gpus+1)*args.epochs
    if args.mode=='pretrain':
        validation_loader=None
    else:
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=args.batch_size,
            num_workers=args.workers, shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True)
    return train_loader,validation_loader

