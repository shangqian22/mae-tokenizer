import torch, os, lightning,time,torchmetrics
import torch.nn as nn
import subprocess
from collections import OrderedDict


class BaseLM(lightning.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.total_steps=args.steps_per_gpu if hasattr(args,'steps_per_gpu') else 1
        self.epochs=args.epochs
        self.lr=args.lr
        self.seed=args.seed
        self.text=vars(args)

    def configure_optimizers(self):
        pass

    def _calculate_loss(self, batch, mode="train"):
        pass
    
    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def log_model(self):
        pass
        
    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def on_train_epoch_start(self):
        if not self.logger == None:
            if not self.logged_gpu:
                train_min=(time.time()-self.start_time)/60
                if train_min>1:
                    self._log_gpu_utilization()
                    self.logged_gpu=True

    def _log_gpu_utilization(self):
        gputil=subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -i=0".split(),text=True)
        self.logger.experiment.add_text( tag='gpu util',text_string=gputil)

    def on_fit_start(self):
        lightning.seed_everything(self.seed, workers=True)
        if not self.logger == None:
            self.logger.log_hyperparams(self.hparams)
            self.logged_gpu=False
            self.log_model()
            for tag,t in self.text.items():
                if not tag in ['data_path','log_dir','max_time','strategy','accelerator']:
                    if not type(t)==str:
                        t=str(t)
                    self.logger.experiment.add_text( tag=tag,text_string=t)
        self.start_time = time.time()

    def on_fit_end(self):
        train_hour=(time.time()-self.start_time)/60**2
        if not self.logger == None:
            self.logger.experiment.add_text( tag='train_hour',text_string=str(train_hour))

        

# In[6]:



    
def get_checkpoint_path(ckpt_dir,full_path=True):
        version= os.listdir(ckpt_dir)[0]
        ckpt= os.listdir(os.path.join(ckpt_dir,version+"/checkpoints"))[0]
        ckpt= version+"/checkpoints/"+ckpt
        if full_path:
            ckpt= os.path.join(ckpt_dir,ckpt)
        return ckpt 

# In[ ]:


def get_resumed(args):
    if args.resume_from_checkpoint:
        args.resumed_ckpt=get_checkpoint_path(
            args.resumed_dir,full_path=False)
        resumed_ckpt=get_checkpoint_path(
            args.resumed_dir)
    else:
        resumed_ckpt=None
    return resumed_ckpt

def load_checkpoint(model, args):
    if args.mode=='finetune' or args.mode=='visualizetion':
        if args.mode=='finetune'or args.mode=='visualization':
            args.ckpt=get_checkpoint_path(
                args.pretrained_dir,full_path=False)
            ckpt_path=get_checkpoint_path(args.pretrained_dir)
        elif args.mode=='validation':
            args.ckpt=get_checkpoint_path(
                args.trained_dir,full_path=False)
            ckpt_path=get_checkpoint_path(args.trained_dir)
        state_dict=torch.load(ckpt_path)['state_dict']
        state_dict = OrderedDict([(key.split("model.")[-1], state_dict[key]) for key in state_dict])
        model.load_state_dict(state_dict,strict=False)
        print('loaded checkpoint for ',args.mode)
    return model
                          
def get_logger(args):
    if not args.log_dir=='':
        save_dir=os.path.join(args.log_dir,args.model_name)
        os.makedirs(save_dir,exist_ok=True)
        return lightning.pytorch.loggers.TensorBoardLogger(save_dir=save_dir)
    else:
        return None
                          
def get_lightning_module(args):
    if 'method1' in args.model_name:
        from method1 import Tokenizer
        from method2 import MAE
        from method1 import ViT
    elif 'method2' in args.model_name:
        from method2 import Tokenizer
        from method2 import MAE
        from method1 import ViT
    elif 'method3' in args.model_name:
        from method3 import Tokenizer
        from model_mae import MAE
        from model_vit import ViT
    if args.mode=='pretrain' or args.mode=='visualization':
        model=MAE(Tokenizer,args)
        if args.mode=='visualization': 
            model=load_checkpoint(model,args)
        from model_mae import PretrainLM as LM
    else:
        model = ViT(Tokenizer,args)
        model=load_checkpoint(model,args)
        if args.mode=='finetune' or args.mode=='validation':
            model=load_checkpoint(model,args)
        if args.mode=='validation':
            args.precision= 32
        if not args.model_name=='yolos-method3':
            from model_vit import FinetuneLM as LM
        else:
            from model_yolos import YOLOS
            model=YOLOS(backbone=model,num_classes=80)
            from model_yolos import FinetuneLM as LM
    lm=LM(model,args)
    return lm
