import sys
sys.path.append('..')
from mae.models_mae import MaskedAutoencoderViT
from util.pe import PositionalEncoding
import torch
import torch.nn as nn
from lightning_modules import BaseLM

class MAE(MaskedAutoencoderViT):
    def __init__(self, Tokenizer,args, **kwargs):
        super().__init__(
            embed_dim=args.embed_dim, depth=args.depth, 
            num_heads=args.num_heads, decoder_embed_dim=args.decoder_embed_dim, 
            decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads,
            mlp_ratio=args.mlp_ratio, norm_layer=args.norm_layer, **kwargs)
        self.embed_dim=args.embed_dim
        self.tokenizer=Tokenizer(args)
        self.mask_ratio=args.mask_ratio
        self.feature_dim=self.tokenizer.resize_size**2*3
        self.embed=torch.nn.Linear(self.feature_dim,self.embed_dim)
        del self.patch_embed
        del self.pos_embed
        self.pos_embed=PositionalEncoding(self.embed_dim)
        del self.decoder_pos_embed
        self.decoder_pos_embed=PositionalEncoding(args.decoder_embed_dim)
        del self.decoder_pred
    
    def forward_encoder(self, x,mask_ratio):
        x=self.embed(x)
        x=self.pos_embed(x)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore
    
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
        x = x[:,1:]
        return x
    
    def forward_loss(self, target, pred, mask):
        loss = (pred - target) ** 2
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs):
        x,ids_patch=self.tokenizer(imgs)
        x, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
        pred = self.forward_decoder(x, ids_restore)
        pred,mask_restored=self.tokenizer.restore(pred,ids_patch,mask)
        loss = self.forward_loss(imgs, pred, mask_restored)
        return loss, pred, mask_restored

class PretrainLM(BaseLM):
    def __init__(self, model,args):
        super().__init__(model,args)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr,
                                      betas=(0.9,0.95), weight_decay=0.05)
        scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                      max_lr=self.lr,
                                                      total_steps=self.total_steps,
                                                      epochs=self.epochs,
                                                      pct_start=0.1)
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
    
    def _calculate_loss(self, batch, mode="train"):
        x,_=batch
        loss, y, mask = self.model(x)
        self.log('train_loss', loss)
        return loss

    def log_model(self):
        self.text.update({'len_keep':str(self.model.tokenizer.len_keep)})
