import torch, PIL, os
import torchvision.transforms as T
import pandas as pd

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self,path,transform,stage='train'):
        self.path=path
        self.stage=stage
        self.transform=transform
        self.preprocess()
        
    def preprocess(self):
        csv=os.path.join(self.path,'LOC_synset_mapping.txt')
        labels=pd.read_csv(csv,sep='\t',header=None)
        syns=labels.iloc[:,0].str[9:]
        labels.columns=['category']
        labels['category']=labels['category'].str[:9]
        labels['syn']=syns
        ids=pd.DataFrame(labels['category'].unique())
        ids.columns=['category']
        ids.reset_index(inplace=True,names='category_id')
        labels=labels.merge(ids,on='category')
        if self.stage=='train':
            csv=os.path.join(self.path,'ILSVRC/ImageSets/CLS-LOC/train_cls.txt')
            train=pd.read_csv(csv, sep=' ',names=['path','sample_id'])
            train['category']=train['path'].str[:9]
            train['path']=os.path.join(self.path,'ILSVRC/Data/CLS-LOC/train/'
                                      )+train['path']+'.JPEG'
            train=train.drop(columns=['sample_id'])
            self.dataset=train.merge(labels,on='category')
        elif self.stage=='validation':
            csv=os.path.join(self.path,'LOC_val_solution.csv')
            validation=pd.read_csv(csv, sep=',')
            validation.columns=['path','category']
            validation['category']=validation['category'].str[:9]
            validation['path']=os.path.join(self.path,'ILSVRC/Data/CLS-LOC/val/'
                                      )+validation['path']+'.JPEG'
            self.dataset=validation.merge(labels,on='category')

    def __getitem__(self, index):
        sample_path=self.dataset['path'][index]
        image = PIL.Image.open(sample_path).convert('RGB')
        sample=self.transform(image)
        target=self.dataset['category_id'][index]
        return sample,target
        
    def __len__(self):
        return(len(self.dataset))
    
def get_imagenet(args):
    path=os.path.join(args.data_path,'imagenet')
    train_transform = T.Compose([
        T.RandomResizedCrop(args.input_size,
            scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset=ImageNetDataset(path=path,
                            transform=train_transform)
    validation_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(args.input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    validation_dataset=ImageNetDataset(path=path,
                                       stage='validation', transform=validation_transform) 
    return train_dataset,validation_dataset