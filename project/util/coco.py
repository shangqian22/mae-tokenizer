MY_CLUSTER_ROOT=''
import os,sys
MY_CLUSTER_ROOT=os.path.expanduser('~/')
sys.path.append(MY_CLUSTER_ROOT+'slurm/ddp')
from YOLOS.datasets.transforms import Compose,RandomHorizontalFlip,RandomSelect,RandomSizeCrop,ToTensor,Normalize
from YOLOS.util.misc import interpolate,_max_by_axis
from YOLOS.util.box_ops import box_xyxy_to_cxcywh
import torch,random,torch,PIL,json
import torchvision.transforms as T
import pandas as pd
import numpy as np

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self,path,transform,stage='train',no_person=False):
        self.path=path
        self.stage=stage
        self.no_person=no_person
        self.transform=transform
        self.preprocess()
        
    def preprocess(self):
        if self.stage=='validation':
            self.stage='val'
        with open(self.path+"/annotations/instances_train2017.json" ) as f:
            file = json.load(f)
        category=pd.DataFrame.from_dict(file['categories'])[['id','name']]
        category.columns=['category_id','category']
        with open(self.path+"/annotations/instances_val2017.json" ) as f:
            file = json.load(f)
        category_val=pd.DataFrame.from_dict(file['categories'])[['id','name']]
        category_val.columns=['category_id','category']
        category=pd.concat([category, category_val]).drop_duplicates()
        category=category.reset_index(names='label')
        
        with open(self.path+"/annotations/captions_%s2017.json" % self.stage) as f:
            file = json.load(f)
        caption=pd.DataFrame.from_dict(file['annotations'])[['id','caption']]
        
        with open(self.path+"/annotations/instances_%s2017.json" % self.stage) as f:
            file = json.load(f)
        dataset=pd.DataFrame.from_dict(file['images'])[['file_name','id']]
        dataset.columns=['path','image_id']
        dataset['path']=self.path+'/%s2017/' % self.stage + dataset['path']
        annotation=pd.DataFrame.from_dict(file['annotations'])[['image_id','bbox','category_id','id','iscrowd']]
        dataset=dataset.merge(annotation,on='image_id')
        dataset=dataset.where(dataset['iscrowd']==0).dropna()
        dataset=dataset.merge(category,on='category_id')
#        dataset=dataset.merge(caption,on='id',how='left')
        dataset=dataset.convert_dtypes()
        self.dataset=dataset
        self.id=dataset['image_id'].unique().dropna()
        if self.no_person:
            self.dataset=self.dataset.where(self.dataset['category']!='person')
        
    def __getitem__(self, index):
        image_id=self.id[index]
        sample_path=self.dataset['path'].where(
            self.dataset['image_id']==image_id).dropna().iloc[0]
        image = PIL.Image.open(sample_path).convert('RGB')
        target=pd.DataFrame(self.dataset[['bbox','label','iscrowd']].where(
            self.dataset['image_id']==image_id).dropna())
        target.columns=['boxes','labels','iscrowd']
        target=target.to_dict(orient='list')
        target['boxes']=torch.as_tensor(target['boxes'])
        target['boxes'][:, 2:] += target['boxes'][:, :2]
        target['labels']=torch.as_tensor(target['labels']).to(torch.int64)
        target['iscrowd']=torch.as_tensor(target['iscrowd']).to(torch.int)
        image,target=self.transform(image,target)
        return image,target
        
    def __len__(self):
        return(len(self.id))
    
class RandomResize:
    def __init__(self, sizes, max_size=None,patch_size=16):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.patch_size=patch_size
        
    def get_size_with_aspect_ratio(self, image_size, size):
        w, h = image_size
        if self.max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > self.max_size:
                size = int(round(self.max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            w_mod = np.mod(w, self.patch_size)
            h_mod = np.mod(h, self.patch_size)
            h = h - h_mod
            w = w - w_mod
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
            ow_mod = np.mod(ow, self.patch_size)
            oh_mod = np.mod(oh, self.patch_size)
            ow = ow - ow_mod
            oh = oh - oh_mod
        else:
            oh = size
            ow = int(size * w / h)
            ow_mod = np.mod(ow, self.patch_size)
            oh_mod = np.mod(oh, self.patch_size)
            ow = ow - ow_mod
            oh = oh - oh_mod
        return (oh, ow)

    def __call__(self, image,target):
        size = random.choice(self.sizes)
        size=self.get_size_with_aspect_ratio(image.size, size)
        rescaled_image = T.functional.resize(image, size=size)
            
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        h, w = size
        target["size"] = torch.tensor([h, w])

        if "masks" in target:
            target['masks'] = interpolate(
                target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5
        return rescaled_image,target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = T.functional.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            target["target_boxes"]=target["boxes"].clone()
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target
    
def get_coco(args):
    path=os.path.join(args.data_path,'coco-2017')
    patch_size=args.patch_size
    scales=(patch_size*(torch.arange(23)+16)).tolist()
    train_transform=Compose([
        RandomHorizontalFlip(),
        RandomSelect(
            RandomResize(scales, max_size=scales[-1] * 1333 // 800,
                         patch_size=patch_size),
            Compose([
                RandomResize([400, 500, 600],
                         patch_size=patch_size),
                RandomSizeCrop(384, 600),
                RandomResize(scales, max_size=scales[-1] * 1333 // 800,
                             patch_size=patch_size),
            ])
        ),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset=CocoDataset(stage='train',path=path, 
                              no_person=args.no_person,
                              transform=train_transform)
    if not args.mode=='pretrain':
        validation_transform=Compose([
            RandomResize([512], max_size=512 * 1333 // 800,
                             patch_size=patch_size),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        validation_dataset=CocoDataset(
            stage='validation',path=path, 
            no_person=args.no_person,
            transform=validation_transform)
    else:
        validation_dataset=None
    return train_dataset,validation_dataset

def collate_fn(batch):        
    batch = list(zip(*batch))    
    return batch
