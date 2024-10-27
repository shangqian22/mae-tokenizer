#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from model_mae import MAE as BaseMAE
from util.pe import PositionalEncoding
from timm.models.vision_transformer import VisionTransformer 

class Tokenizer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.fold_size=args.fold_size
        self.folds=args.folds
        self.min_len_keep=args.min_len_keep
        self.levels=args.levels
        self.len_keep_level=torch.tensor([args.len_keep_level]*self.levels)
        self.len_keep=self.len_keep_level.sum()
        self.scale=args.scale
        self.vocab_size=self.scale**4+1
        self.register_buffer("base", self.scale ** torch.arange(4).flip([0]), persistent=False)
        
    def patchify(self,x):
        B,C,H,W = x.shape
        H=H-H%self.fold_size
        W=W-W%self.fold_size
        x=x[:,:,:H,:W].clone()
        patches_h=H//self.fold_size
        patches_w=W//self.fold_size
        patch = x.reshape(B,C, patches_h, self.fold_size, 
                          patches_w, self.fold_size)
        patch = patch.permute(0, 2, 4,1,3,5)
        patch = patch.flatten(1,2)
        return patch,patches_h,patches_w

    def get_directions(self, x):
        N=x.size(0)
        directions=torch.zeros(N,4,device=x.device)
        patch_size=self.fold_size
        half=patch_size//2
        square_triangle=(patch_size**2-patch_size)/2
        square=patch_size**2/2

        #left upper
        patches=x
        a=torch.triu(patches)
        b=torch.tril(patches)
        b=torch.flip(b,dims=[1]).rot90(-1,[1,2])
        directions[:,0]=torch.abs(a-b).sum([1,2])/square_triangle
        # right upper
        d1=patches.rot90(-1,[1,2])
        a=torch.triu(d1)
        b=torch.tril(d1)
        b=torch.flip(b,[1]).rot90(-1,[1,2])
        directions[:,1]=torch.abs(a-b).sum([1,2])/square_triangle
        # h
        a=patches[:,:,:half]
        b=patches[:,:,half:]
        directions[:,2]=torch.abs(a-b).sum([1,2])/square
        # v
        a=patches[:,:half,:]
        b=patches[:,half:,:]
        directions[:,3]=torch.abs(a-b).sum([1,2])/square

        direction=torch.argmin(directions,dim=-1)
        direction_one_hot=nn.functional.one_hot(direction, num_classes=4)
        return direction,direction_one_hot
    

    def fold(self, direction,one_hot):
        padded=nn.functional.pad(direction,(1,1,1,1),"constant",4)
        s7=padded[:,:-2,:-2]
        s8=padded[:,:-2,1:-1]
        s9=padded[:,:-2,2:]
        s4=padded[:,1:-1,:-2]
        s6=padded[:,1:-1,2:]
        s1=padded[:,2:,:-2]
        s2=padded[:,2:,1:-1]
        s3=padded[:,2:,2:]

        scope=one_hot.sum(-1)
        one_hot[:,:,:,0]+=scope*((direction-s9)==0)*(direction!=4)
        one_hot[:,:,:,0]+=scope*((direction-s1)==0)*(direction!=4)
        one_hot[:,:,:,1]+=scope*((direction-s7)==0)*(direction!=4)
        one_hot[:,:,:,1]+=scope*((direction-s3)==0)*(direction!=4)
        one_hot[:,:,:,2]+=scope*((direction-s8)==0)*(direction!=4)
        one_hot[:,:,:,2]+=scope*((direction-s2)==0)*(direction!=4)
        one_hot[:,:,:,3]+=scope*((direction-s4)==0)*(direction!=4)
        one_hot[:,:,:,3]+=scope*((direction-s6)==0)*(direction!=4)
        return one_hot

    @torch.no_grad()
    def forward(self, x):
        x,H,W=self.patchify(x)
        B=x.size(0)
        
        x=x.mean(2).flatten(0,1)
        direction,direction_one_hot=self.get_directions(x)
        direction=direction.view(B,H,W)
        direction_one_hot=direction_one_hot.view(B,H,W,4)
        
        for f in range(self.folds):
            direction_one_hot=self.fold(direction,direction_one_hot)
        scope=direction_one_hot.sum(-1)
        scope=scope.flatten(1,2)
        direction_one_hot=direction_one_hot.flatten(1,2)
        
        center_cord=torch.stack([
            scope.max(dim=-1).indices//H, 
            scope.max(dim=-1).indices%H])
        center_cord=center_cord.transpose(0,1).unsqueeze(1)

        cord_h=torch.arange(H,device=x.device).unsqueeze(1).repeat(1,W)
        cord_w=torch.arange(W,device=x.device).unsqueeze(0).repeat(H,1)
        cord=torch.stack([cord_h,cord_w],dim=2).flatten(0,1)
        cord=cord.unsqueeze(0).repeat(B,1,1)
        distance=((center_cord-cord)**2).sum(-1)
        index=torch.argsort(distance, dim=-1,descending=True)
        direction_one_hot=torch.gather(direction_one_hot,1,
               index.unsqueeze(-1).expand(-1,-1,4))
        
        token=direction_one_hot/direction_one_hot.sum(-1,keepdim=True)
        token=(token*self.scale).round()
        token= torch.minimum(token, torch.tensor(self.scale))
        len_trim=token.size(1)-token.size(1)%self.levels
        token=token[:,:len_trim]
        token=token.reshape(B,self.levels,-1,4)
        token=torch.sum(self.base*token,-1).to(torch.long)
        keep_mask=torch.zeros(*token.shape,dtype=torch.bool,device=x.device)
        for i,l in enumerate(self.len_keep_level):
            keep_mask[:,i,:l]=1
        token=torch.masked_select(token,keep_mask)
        token=token.reshape(B,self.len_keep)
        return token,None


class MAE(BaseMAE):
    def __init__(self, Tokenizer, args, **kwargs):
        super(BaseMAE, self).__init__(
            embed_dim=args.embed_dim, depth=args.depth, 
            num_heads=args.num_heads, decoder_embed_dim=args.decoder_embed_dim, 
            decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads,
            mlp_ratio=args.mlp_ratio, norm_layer=args.norm_layer, **kwargs)
        self.embed_dim=args.embed_dim
        self.tokenizer=Tokenizer(args)
        self.mask_ratio=args.mask_ratio
        self.decoder_embed_dim=args.decoder_embed_dim
        self.embed = nn.Embedding(self.tokenizer.vocab_size, 
                                  self.embed_dim)
        del self.patch_embed
        del self.pos_embed
        self.pos_embed=PositionalEncoding(self.embed_dim)
        del self.decoder_pos_embed
        self.decoder_pos_embed=PositionalEncoding(self.decoder_embed_dim)
        del self.decoder_pred
        self.decoder_pred=nn.Linear(self.decoder_embed_dim,self.tokenizer.vocab_size)
        self.loss_fct = nn.CrossEntropyLoss()
    
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        x_p,x=x[:,:1,:],x[:,1:]
        
        B,N,F=x.shape
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - N, 1)                              
        x_ = torch.cat([x, mask_tokens], dim=1)  
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, F))
        x=self.decoder_pos_embed(x)
        
        x = torch.cat([x_p, x], dim=1)  
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:,1:]
        return x
    
    def forward(self, imgs):
        token,_=self.tokenizer(imgs)
        x, mask, ids_restore = self.forward_encoder(token, self.mask_ratio)
        pred = self.forward_decoder(x, ids_restore)
        loss = self.forward_loss(token, pred, mask)
        return loss, pred, mask
    
    def forward_loss(self, target,pred, mask):
        B=mask.size(0)
        mask=mask.to(torch.bool)
        target=torch.masked_select(target,mask)
        idx=torch.nonzero(mask,as_tuple=True)[
                1].reshape(B,-1).unsqueeze(-1).expand(
                        -1,-1,self.tokenizer.vocab_size)
        pred=torch.gather(pred,1,idx)
        pred=pred.flatten(0,1)
        loss=self.loss_fct(pred,target)
        return loss
