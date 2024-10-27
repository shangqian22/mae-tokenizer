# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from model_mae import MAE as BaseMAE
from util.pe import PositionalEncoding
from timm.models.vision_transformer import VisionTransformer 

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
        self.decoder_pred=nn.Linear(self.decoder_embed_dim,self.tokenizer.predict_dim)
        
    
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
        x,ids_patch=self.tokenizer(imgs)
        x, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
        pred = self.forward_decoder(x, ids_restore)
        pred,mask_restored=self.tokenizer.restore( pred,ids_patch,mask)
        loss = self.forward_loss(imgs.mean(1,keepdim=True), pred, mask_restored)
        return loss, pred, mask_restored
    
# In[4]:
class ViT(VisionTransformer):
    def __init__(self, Tokenizer,
                 args, **kwargs):
        super().__init__(embed_dim=args.embed_dim, global_pool='avg',**kwargs)
        self.tokenizer=Tokenizer(args)
        self.embed_dim=args.embed_dim
        del self.patch_embed
        self.embed = nn.Embedding(self.tokenizer.vocab_size, self.embed_dim)
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

class Tokenizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        min_keep_ratio=args.min_keep_ratio
        self.bits=12
        self.input_dim=args.input_size
        self.levels=args.levels
        n=torch.arange(0,self.levels)
        self.patches_axis=torch.as_tensor([2]*self.levels)**(n+1)
        self.patches0=self.patches_axis**2
        self.patches1=(self.patches_axis-1)**2
        self.patches_level=self.patches0+self.patches1
        self.patch_size=self.input_dim//self.patches_axis
        self.patch_size_shift=self.patch_size//2
        self.band_size=self.patch_size//6
        self.len_max=self.patches_level.sum()
        keep_ratio=1-(1-min_keep_ratio)/(self.levels-1)*torch.arange(0,self.levels)
        self.len_keep_level=torch.round(self.patches_level*keep_ratio).to(torch.int)
        self.len_keep=self.len_keep_level.sum()
        self.len_pad=self.len_max-self.len_keep
        last_index_level=self.patches_level.cumsum(dim=0)
        self.index_level=torch.stack([last_index_level-self.patches_level,
                                 last_index_level],dim=0).transpose(0,1)
        self.register_buffer("base",2 ** torch.arange(self.bits).flip([0]),persistent=False)
        first_index_level=self.patches_level.cumsum(dim=0)-self.patches_level
        last_index_level=first_index_level+self.len_keep_level
        index_level=torch.stack([first_index_level,
                                 last_index_level],dim=0).transpose(0,1)
        self.register_buffer("keep_mask", torch.zeros(
            1,self.len_max,
            dtype=torch.bool),persistent=False)
        for start,end in index_level:
            self.keep_mask[:,start:end]=1
        self.register_buffer("token", torch.zeros(
            1,self.len_max,
            dtype=torch.long),persistent=False)
        self.vocab_size=2**self.bits
        
        self.predict_dim=3*self.bits
        patch_segment=[]
        for patches_level,band_size in zip(self.patches_level,self.band_size):
            patch_segment.append(torch.ones(1,patches_level,1,6,6*band_size*band_size,dtype=torch.int))
        for i,s in enumerate(patch_segment):
            self.register_buffer(f"patch_segment_{i}", s)
        self.register_buffer("token_map", torch.zeros(
            1,1,self.input_dim,self.input_dim,
            dtype=torch.int),persistent=False)
        self.register_buffer("restored", torch.zeros(
            1,1,self.input_dim,self.input_dim), persistent=False)
        self.register_buffer("pad", torch.zeros(
            1,1,1), persistent=False)
    @property
    def patch_segment(self):
        return [getattr(self, f"patch_segment_{l}") for l in range(self.levels)]
        
    def patchify(self,x,patches_axis,patch_size):
        B,C = x.shape[:2]
        patch = x.reshape(B,C, patches_axis, patch_size, patches_axis, patch_size)
        patch = patch.permute(0, 2, 4,1,3,5)
        patch = patch.flatten(1, 2)
        return patch
        
    def overlapped_patchify(self,x,i):
        patch0=self.patchify(x,self.patches_axis[i],self.patch_size[i])
        end=self.input_dim-self.patch_size_shift[i]
        x_shift=x[:,:,self.patch_size_shift[i]:end,self.patch_size_shift[i]:end]
        patch1=self.patchify(x_shift,self.patches_axis[i]-1,self.patch_size[i])
        patch=torch.cat([patch0,patch1],dim=1)
        return patch
        
    def token2patch(self,token_raw,level):
        B,L,C=token_raw.shape[:3]
        band_size=self.band_size[level]
        patches_level=self.patches_level[level]
        patch_size=self.patch_size[level]
        patch_segment=self.patch_segment[level].expand(-1,-1,C,-1,-1)
        v=patch_segment*token_raw[:,:,:,:6].unsqueeze(4)
        h=patch_segment*token_raw[:,:,:,6:].unsqueeze(4)
        v=v.reshape(B,patches_level,C,6,6,band_size,band_size
                     ).permute(0,1,2,3,5,4,6
                     ).reshape(B,patches_level,C,patch_size,patch_size)
        h=h.reshape(B,patches_level,C,6,6,band_size,band_size
                     ).permute(0,1,2,4,5,3,6
                     ).reshape(B,patches_level,C,patch_size,patch_size)
        return v,h
    
    def unpatchify(self,patch,patches_axis,patch_size,shift=False):
        B,L,C=patch.shape[:3]
        input_dim=self.input_dim-(patch_size)*shift
        patch=patch.view(B,patches_axis,patches_axis,C,patch_size,patch_size
                     ).permute(0,3,1,4,2,5
                     ).reshape(B,C,input_dim,input_dim)
        return patch
        
    def remap_token(self,token_raw,token_map,level):
        v,h=self.token2patch(token_raw,level)
        v0,v1=v.split([self.patches0[level],self.patches1[level]],dim=1)
        h0,h1=h.split([self.patches0[level],self.patches1[level]],dim=1)
        patches_axis=self.patches_axis[level]
        patch_size=self.patch_size[level]
        v0=self.unpatchify(v0,patches_axis,patch_size)
        v1=self.unpatchify(v1,patches_axis-1,patch_size,shift=True)
        h0=self.unpatchify(h0,patches_axis,patch_size)
        h1=self.unpatchify(h1,patches_axis-1,patch_size,shift=True)
        token_map+=v0
        token_map+=h0
        end=self.input_dim-self.patch_size_shift[level]
        token_map[:,:,self.patch_size_shift[level]:end,self.patch_size_shift[level]:end]+=v1
        token_map[:,:,self.patch_size_shift[level]:end,self.patch_size_shift[level]:end]+=h1
        return token_map
        
    def patch2token(self,patch,patches_level,band_size):
        B,L,C=patch.shape[:3]
        quantile=torch.median(patch.flatten(2,4), dim=2,keepdim=True)[0]
        patch = patch.reshape(B,patches_level,C, 6, band_size, 6, band_size)
        patch = patch.permute(0,1,3,5,2,4,6)  # B,N, 6,6, C,H,W
        v=(patch.flatten(3,6).mean(3)<quantile)
        h=(patch.transpose(2,3).flatten(3,6).mean(3)<quantile)
        token_raw=torch.cat([v,h],dim=2) #B,N,12
        return token_raw
    
    def gather_token(self,token,token_map):
        B= token.size(0)
        score=self.token.expand(B,-1).clone()
        q2=torch.median(token_map.flatten(2,3),-1,keepdims=True)[0]
        for l in range(self.levels):
            level_score=self.overlapped_patchify(token_map,l).flatten(2,4)
            level_score=torch.abs(level_score-q2).sum(-1)
            score[:,self.index_level[l][0]:self.index_level[l][1]]=level_score
        index=torch.argsort(score, dim=1,descending=True)
        ids_restore=torch.argsort(index)
        mask=self.keep_mask.expand(B,-1).clone()
        token=torch.masked_select(token,mask)
        token=token.reshape(B,-1)
        return token,ids_restore
        
    def forward(self, x):
        B= x.size(0)
        token=self.token.expand(B,-1).clone()
        token_map=self.token_map.expand(B,-1,-1,-1).clone()
        for l in range(self.levels):
            patch =self.overlapped_patchify(x,l)
            token_raw=self.patch2token(patch,self.patches_level[l],self.band_size[l])
            token_map=self.remap_token(token_raw.unsqueeze(2),token_map,l)
            token_level=torch.sum(self.base*token_raw,-1)
            token[:,self.index_level[l][0]:self.index_level[l][1]]=token_level
        return self.gather_token(token,token_map)
    
    def restore(self,token_raw,ids_restore,mask):
        B=token_raw.size(0)
        pad=self.pad.expand(B,self.len_pad,self.predict_dim)
        token_raw=torch.cat([token_raw, pad],dim=1)
        ids=ids_restore.unsqueeze(-1).expand(-1,-1,self.predict_dim)
        token_raw=torch.gather(token_raw,1,ids)
        token_raw=token_raw.reshape(B,self.len_max,3,self.bits)
        
        mask=mask.unsqueeze(-1).expand(-1,-1,self.bits)
        mask_pad=self.pad.expand(B,self.len_pad,self.bits)
        mask=torch.cat([mask, mask_pad],dim=1)
        ids=ids_restore.unsqueeze(-1).expand(-1,-1,self.bits)
        mask=torch.gather(mask,1,ids)
        
        restored=self.restored.expand(B,3,-1,-1).clone()
        mask_restored=self.restored.expand(B,-1,-1,-1).clone()
        for i in range(self.levels):
            token_level=token_raw[:,self.index_level[i][0]:self.index_level[i][1]]
            restored=self.remap_token(token_level,restored,i)
        
            mask_level=mask[:,self.index_level[i][0]:self.index_level[i][1]]
            mask_level=mask_level.unsqueeze(2)
            mask_restored=self.remap_token(mask_level,mask_restored,i)
        restored/=self.levels
        return restored,mask_restored
