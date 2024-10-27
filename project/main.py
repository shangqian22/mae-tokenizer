#!/usr/bin/env python
# coding: utf-8

# In[1]:
MY_CLUSTER_ROOT=''
import os
MY_CLUSTER_ROOT=os.path.expanduser('~/')


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


# In[2]:

import sys
if is_interactive():
    sys.path.append(MY_CLUSTER_ROOT+'slurm/ddp/project')


# In[3]:


from util.misc import add_model_args,get_dataloader
from lightning_modules import get_lightning_module,load_checkpoint,get_logger,get_resumed
import argparse,lightning


# In[4]:


def get_args_parser():
    parser = argparse.ArgumentParser('ViT pretraining and finetuning script')
    parser.add_argument('--model_name', type=str,  default='')
    parser.add_argument('--mode', default='pretrain')
    parser.add_argument('--input_size', default=256,type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--base_lr', type=float, default=-1)
    parser.add_argument("--data_path", type=str, default=MY_CLUSTER_ROOT+'slurm/data')
    parser.add_argument('--gpus', default=4,type=int)
    parser.add_argument('--nodes', default=2,type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument("--log_dir", type=str, default=MY_CLUSTER_ROOT+'slurm/log')
    parser.add_argument("--pretrained_dir", type=str, default=MY_CLUSTER_ROOT+'slurm/ddp/project/pretrained')
    parser.add_argument("--trained_dir", type=str, default=MY_CLUSTER_ROOT+'slurm/ddp/project/trained')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--resume_from_checkpoint', default=False, action='store_true')
    parser.add_argument("--resumed_dir", type=str, default=MY_CLUSTER_ROOT+'slurm/ddp/project/resumed')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default='16-mixed')
    parser.add_argument("--strategy", type=str, default='ddp')
    parser.add_argument("--accelerator", type=str, default='gpu')
    parser.add_argument("--max_time", type=str, default=None)
    return parser


# In[5]:


if __name__ == "__main__":
    if is_interactive():
        params="--model_name=yolos-method3 --resize_size=32 --min_keep_ratio=0.3 --input_size=256 --epochs=1 --batch_size=1 --lr=1e-5 --mode=from_scratch".split()
        params+=" --nodes=1 --gpus=1 --precision=32 --strategy=auto --accelerator=cpu".split()
    else:
        params=None
    parser = get_args_parser()
    parser=add_model_args(params,parser)
    args = parser.parse_args(params)
    if args.lr==-1 and not args.base_lr==-1:
        args.lr=args.base_lr*args.gpus*args.batch_size/256
    
    train_loader,validation_loader=get_dataloader(args)
    lightning_module=get_lightning_module(args)
    tb_logger=get_logger(args)
    resumed_ckpt=get_resumed(args)
    callbacks=[
        lightning.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
        lightning.pytorch.callbacks.ModelCheckpoint(
            every_n_train_steps=min(1000,int(args.steps_per_gpu*0.01))),
        ]
    trainer = lightning.Trainer(accelerator=args.accelerator, 
                        strategy=args.strategy,num_nodes=args.nodes, 
                        max_time=args.max_time, max_epochs=args.epochs,
                        logger=tb_logger, precision=args.precision, 
                        enable_progress_bar=False,
                        profiler="simple",
                        callbacks=callbacks)
    if args.mode=='pretrain':
        trainer.fit(lightning_module, train_loader,ckpt_path=resumed_ckpt)
    elif not args.mode=='validation':
        trainer.fit(lightning_module, train_loader,validation_loader,ckpt_path=resumed_ckpt)
    else:
        trainer.validate(lightning_module, validation_loader,ckpt_path=resumed_ckpt)

