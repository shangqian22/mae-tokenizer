#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from timm.models.vision_transformer import VisionTransformer 
from util.pe import PositionalEncoding
import torch,torchmetrics
import torch.nn as nn
from functools import partial
from lightning_modules import BaseLM
from torch.jit import Final
from timm.layers import Mlp,use_fused_attn
from typing import Optional
import torch.nn.functional as F


# In[ ]:


class ViT(VisionTransformer):
    def __init__(self, Tokenizer,
                 args, **kwargs):
        super().__init__(embed_dim=args.embed_dim, global_pool='avg',**kwargs)
        self.tokenizer=Tokenizer(args)
        self.embed_dim=args.embed_dim
        del self.tokenizer.pred_features
        del self.patch_embed
        self.embed=torch.nn.Linear(self.tokenizer.feature_dim,self.embed_dim)
        del self.pos_embed
        self.pos_embed=PositionalEncoding(self.embed_dim)

    def forward_features(self, x):
        B = x.shape[0]
        x,_=self.tokenizer(x)
        x=self.embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x=self.pos_embed(x)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x


# In[ ]:


class FinetuneLM(BaseLM):
    def __init__(self, model, args):
        super().__init__(model,args)
        self.criterion = nn.CrossEntropyLoss()        
        self.metric = torchmetrics.Accuracy(task="multiclass", 
                                         num_classes=self.model.num_classes)
        self.metric5 = torchmetrics.Accuracy(task="multiclass", 
                                         num_classes=self.model.num_classes,top_k=5)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr,
                                      weight_decay=0.05)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                      max_lr=self.lr,
                                                      total_steps=self.total_steps,
                                                      epochs=self.epochs,
                                                      pct_start=0.1)
        return [optimizer], {"scheduler": scheduler, "interval": "step"}

    def _calculate_loss(self, batch, mode="train"):
        x,y=batch
        preds=self.model(x)
        loss=self.criterion(preds,y) 
        if not mode=='val':
            self.log("%s_loss" % mode, loss)
        else:
            self.log("%s_loss" % mode, loss,on_step=True, on_epoch=True, sync_dist=True)
            accuracy1=self.metric(preds, y)
            accuracy5=self.metric5(preds, y)
            self.log("%s_acc_top1" % mode, accuracy1,on_step=True, on_epoch=True, sync_dist=True)
            self.log("%s_acc_top5" % mode, accuracy5,on_step=True, on_epoch=True, sync_dist=True)
        return loss


# In[ ]:





# In[3]:


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B, num_head, N, c
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return x, attn
        else:
            return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=return_attention)
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


# In[ ]:


class ViT_return_Attention(nn.Module):
    def __init__(self, model,args):
        super().__init__()
        state_dict=model.state_dict()
        del model.blocks
        model.blocks = nn.Sequential(*[
            Block( dim=args.embed_dim,
                  num_heads=args.num_heads,
                  mlp_ratio=args.mlp_ratio,
                  qkv_bias=args.qkv_bias,
                  norm_layer=args.norm_layer )
            for i in range(args.depth)])
        model.load_state_dict(state_dict)
        self.model=model

    def forward(self, x):
        B = x.size(0)
        x,_=self.model.tokenizer(x)
        x=self.model.embed(x)

        cls_tokens = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x=self.model.pos_embed(x)
        x = self.model.pos_drop(x)

        output = []
        for i in range(len((self.model.blocks))):
            x, attn = self.model.blocks[i](x, return_attention=True)

            if i == len(self.model.blocks)-1:
                output.append(attn)
        return output

