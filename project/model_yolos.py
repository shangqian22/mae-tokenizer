#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os,sys
sys.path.append('..')
from timm.models.layers import trunc_normal_
import torch.nn as nn
import torch
from lightning_modules import BaseLM 
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from YOLOS.models.detector import MLP,PostProcess
from YOLOS.models.matcher import HungarianMatcher
from YOLOS.models.detector import SetCriterion

class YOLOS(nn.Module):
    def __init__(self, backbone,num_classes=80, det_token_num=100, 
                 init_pe_size=[800,1344], mid_pe_size=None, 
                ):
        super().__init__()
        del backbone.fc_norm
        del backbone.head
        self.backbone=backbone
        self.det_token_num = det_token_num
        self.det_token = nn.Parameter(torch.zeros(1, det_token_num, self.backbone.embed_dim))
        self.det_token = trunc_normal_(self.det_token, std=.02)
        
        hidden_dim=self.backbone.embed_dim
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
    
    def forward(self, x):
        x=self.backbone_forward1(self.backbone,x)
        
        B = x.size(0)
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((x, det_token), dim=1)
        
        x=self.backbone_forward2(self.backbone,x)
        
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out
    
    def backbone_forward1(self,backbone,x):
        B = x.shape[0]
        
        x=backbone.embed(x)

        cls_tokens = backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x=backbone.pos_embed(x)
        return x
    
    def backbone_forward2(self,backbone,x):
        for blk in backbone.blocks:
            x = blk(x)
            
        return x[:, -self.det_token_num:, :]
    

# In[ ]:


class FinetuneLM(BaseLM):
    def __init__(self, model,args):
        super().__init__(model,args)
        matcher = HungarianMatcher()
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5,'loss_giou': 2}
        losses= ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(num_classes=80, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=0.1, losses=losses)
        self.postprocess=PostProcess()
        self.map = MeanAveragePrecision()
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer_backbone = torch.optim.AdamW(self.model.backbone.parameters(),lr=self.lr/10,
                                      weight_decay=1e-4)
        parameters=(i[1] for i in filter(lambda x: not 'backbone' in x[0],  self.model.named_parameters() ))
        optimizer = torch.optim.AdamW(parameters,lr=self.lr,
                                      weight_decay=1e-4)
        scheduler_backbone=torch.optim.lr_scheduler.OneCycleLR(optimizer_backbone,
                                                      max_lr=self.lr/5,
                                                      total_steps=self.total_steps,
                                                      epochs=self.epochs,
                                                      pct_start=0.1)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                      max_lr=self.lr,
                                                      total_steps=self.total_steps,
                                                      epochs=self.epochs,
                                                      pct_start=0.1)
        return [optimizer_backbone,optimizer],  [
            {"scheduler": scheduler_backbone, "interval": "step"},
            {"scheduler": scheduler, "interval": "step"}]
    
    
    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        
        loss = self._calculate_loss(batch, mode="train")
        
        opt1.zero_grad()
        opt2.zero_grad()
        self.manual_backward(loss)
        opt1.step()
        opt2.step()

        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

    def _calculate_loss(self, batch, mode="train"):
        x,y=batch
        
        B = len(x)
        tensor=torch.zeros(B,self.model.backbone.tokenizer.len_keep,
                           self.model.backbone.tokenizer.feature_dim,device=self.device)
        for i in range(B):
            token,_=self.model.backbone.tokenizer(x[i].unsqueeze(0))
            tensor[i]=token[0]
        batch_size=tensor.size(0)
            
        out=self.model(tensor)
        
        loss_dict = self.criterion(out, y)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        if not mode=='val':
            self.log("%s_loss" % mode, loss,batch_size=batch_size)
            for k in loss_dict.keys():
                if k in weight_dict:
                    self.log("loss_%s" % k,loss_dict[k],batch_size=batch_size)
        else:
            self.log("%s_loss" % mode, loss,on_step=True, on_epoch=True,
                     sync_dist=True,batch_size=batch_size)
            size=torch.stack([i['size'] for i in y])
            pred=self.postprocess(out,size)
            for t in y:
                t['boxes']=t['target_boxes']
            accuracy=self.map(pred, y)
            self.log("%s_map" % mode, accuracy['map'].to(self.device),on_step=True, on_epoch=True, sync_dist=True,batch_size=batch_size)
            self.log("%s_map50" % mode, accuracy['map_50'].to(self.device),on_step=True, on_epoch=True, sync_dist=True,batch_size=batch_size)
            self.log("%s_map75" % mode, accuracy['map_75'].to(self.device),on_step=True, on_epoch=True, sync_dist=True,batch_size=batch_size)
        return loss

