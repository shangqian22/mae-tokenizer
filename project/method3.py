#!/usr/bin/env python
# coding: utf-8

# In[18]:


import torch
import torch.nn as nn
import torchvision.transforms as T
class Tokenizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim=args.input_size
        self.levels=args.levels
        self.resize_size=args.resize_size
        n=torch.arange(0,self.levels)
        self.patches_axis=torch.as_tensor([2]*self.levels)**(n+1)
        self.patch_size_level=self.input_dim//self.patches_axis
        self.patches_level=self.patches_axis**2
        self.len_max=self.patches_level.sum()
        keep_ratio=1-(1-args.min_keep_ratio)/(self.levels-1)*torch.arange(0,self.levels)
        self.len_keep_level=torch.round(self.patches_level*keep_ratio).to(torch.int)
        self.len_keep=self.len_keep_level.sum()
        
        self.resize=T.Resize(self.resize_size)
        self.feature_dim=self.resize_size**2*3
        self.register_buffer("gathered", torch.zeros(
            1,self.len_keep, self.feature_dim),
                             persistent=False)
        self.register_buffer("ids_restore", torch.zeros(
            self.levels,1,1,
            dtype=torch.int64), persistent=False)
        last_index_level=self.len_keep_level.cumsum(dim=0)
        self.index_level=torch.stack([last_index_level-self.len_keep_level,
                                 last_index_level],dim=0).transpose(0,1)
        
        self.crop_size_level=self.input_dim//self.patches_axis
        self.len_pad_level=self.patches_level-self.len_keep_level
        self.register_buffer("pad", torch.zeros(
            1,1,1), persistent=False)
        self.pred_features=nn.ModuleList([nn.Linear(args.decoder_embed_dim,crop_size**2*3)
                                          for crop_size in self.crop_size_level])
        self.register_buffer("restored", torch.zeros(
            1,3,self.input_dim,self.input_dim), persistent=False)
                 
    def gather_feature(self,feature,i):
        patches_level=feature.size(1)
        entropy = -torch.sum(feature * torch.log(feature),dim=-1) / torch.log(torch.as_tensor(torch.e))
        index=torch.argsort(entropy,dim=1,descending=False)
        ids_restore=torch.argsort(index)[:,:patches_level]
        feature=torch.gather(feature,1, index[:,:self.len_keep_level[i]
                         ].unsqueeze(-1).expand(-1,-1,self.feature_dim))
        return feature, ids_restore
        
    def patchify(self,x,i):
        B,C,H,W = x.shape
        patch_size_level=self.patch_size_level[i]
        H=H-H%patch_size_level
        W=W-W%patch_size_level
        x=x[:,:,:H,:W].clone()
        patch = x.reshape(B,C, H//patch_size_level, patch_size_level, 
                          W//patch_size_level, patch_size_level)
        patch = patch.permute(0, 2, 4,1,3,5)
        patch = patch.flatten(1, 2)
        return patch
    
    def forward(self, x):
        B, C, H, W = x.shape
        gathered=self.gathered.expand(B,-1,-1).clone()
        max_ids=(max(H,W)//self.patch_size_level[-1])**2
        ids_restore=self.ids_restore.expand(-1,B,max_ids).clone()
        for i in range(self.levels):
            patch = self.patchify(x,i)
            patches_level=patch.size(1)
            
            patch=patch.flatten(0,1)
            patch=self.resize(patch)
            feature = patch.flatten(1,3) 
            feature=feature.reshape(B,patches_level,self.feature_dim)

            feature,ids_restore_level =self.gather_feature(feature,i)
            gathered[:,self.index_level[i][0]:self.index_level[i][1]]=feature
            ids_restore[i,:,:patches_level]=ids_restore_level
        return gathered,ids_restore
    
    def unpatchify(self,patch,patches_axis,patch_size):
        B,L,C=patch.shape[:3]
        patch=patch.view(B,patches_axis,patches_axis,
             C,patch_size,patch_size).permute(0,3,1,4,2,5
             ).reshape(B,C,self.input_dim,self.input_dim)
        return patch
    
    def restore(self,gathered,ids_restore,mask):
        B=gathered.size(0)
        restored=self.restored.expand(B,-1,-1,-1).clone()
        mask_restored=restored.clone()
        for i in range(self.levels):
            feature=gathered[:,self.index_level[i][0]:self.index_level[i][1]]
            patch=self.pred_features[i](feature)
            D=patch.size(-1)
            pad=self.pad.expand(B,self.len_pad_level[i],D)
            patch=torch.cat([patch, pad],dim=1)
            
            ids=ids_restore[i,:,:self.patches_level[i]]
            ids=ids.unsqueeze(-1).expand(-1,-1,D)
            patch=torch.gather(patch,1,ids)
            
            crop_size=self.crop_size_level[i]
            patch=patch.view(B,self.patches_level[i],
                                 3,crop_size,crop_size)
            
            restored+=self.unpatchify(patch,self.patches_axis[i],
                                    self.patch_size_level[i])
        
            m=mask[:,self.index_level[i][0]:self.index_level[i][1]]
            m=m.unsqueeze(-1).expand(-1,-1,D)
            
            patch=torch.cat([m, pad],dim=1)
            patch=torch.gather(patch,1,ids)
            patch=patch.view(B,self.patches_level[i],
                                 3,crop_size,crop_size)
            
            mask_restored+=self.unpatchify(patch,self.patches_axis[i],
                                    self.patch_size_level[i])
        restored/=self.levels
        return restored,mask_restored

