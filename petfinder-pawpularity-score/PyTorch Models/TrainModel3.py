import sys 
import os
import math
import gc
from utils import colorstr, LOGGER
sys.path.append(os.path.abspath("pytorch-image-models"))
from timm import create_model
import pandas as pd
import cv2
import random 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from mydataloader import MyDataGenerator
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from albumentations import (Compose, Rotate, HorizontalFlip, RandomBrightnessContrast, Resize, 
                            ChannelShuffle, VerticalFlip, RandomCrop, HueSaturationValue)
from albumentations.augmentations.transforms import Blur, CoarseDropout
from fastai.vision.all import BCEWithLogitsLossFlat

seed=365
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True


class CustomDataGenerator(torch.utils.data.Dataset):
    
    def __init__(self, df, dim, train_flag = True):
        super(CustomDataGenerator, self).__init__()
        'Initialization'
        self.df = df
        self.dim = dim
        self.train_flag = train_flag

        # define data augmentation for train 
        self.transform_train = Compose([
            # rotate 
            Rotate(limit=20, p=random.uniform(0.4, 0.6)),
            # horizontal flip
            HorizontalFlip(p=random.uniform(0.5, 0.7)),
            # vertical flip
            VerticalFlip(p=random.uniform(0.2, 0.3)),
            # random brigthtness
            RandomBrightnessContrast(p=random.uniform(0.4, 0.6)),
            # channel shuffle
            ChannelShuffle(p=np.random.uniform(0.3, 0.5)),
            HueSaturationValue(),
            Blur(always_apply=False, p=0.5, blur_limit=(3, 7)),
            CoarseDropout(always_apply=False, p=0.5, max_holes=12, max_height=60, max_width=60, 
                          min_holes=8, min_height=20, min_width=20),
            Resize(always_apply=True, 
                   height=self.dim[0], 
                   width=self.dim[1], 
                   interpolation=0)
            ])
        # define augmentation for validation 
        self.transform_val = Compose([
            Resize(always_apply=True, 
                   height=self.dim[0], 
                   width=self.dim[1], 
                   interpolation=0)
            ])
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.df)
    
       
    
    def __getitem__(self, idx):
        'Generates data containing batch_size samples'       
        # generate data :
        img = cv2.imread(self.df.iloc[idx, 0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, self.dim[:2])
        img = np.array(img, dtype='uint8')
        # data augmentation 
        img = self.__image_augmentation(img)
        img = img / 255.0 
        # transpose axis 
        img = img.transpose(2,1,0)
        
        meta = np.array(self.df.iloc[idx, 1:-3].values, dtype='float32')
        y = self.df.iloc[idx, -2]
            
        return torch.tensor(img), torch.tensor(meta),  torch.tensor(y)
    
    def __image_augmentation(self, img):
        if self.train_flag:
            return self.transform_train(image=img)['image']
        else:
            return self.transform_val(image=img)['image']


class CheckpointCallback():
    def __init__(self,model_filename, mode='max',verbose=0) :
        self.model_filename=model_filename
        self.mode=mode
        self.verbose=verbose
        if(mode=='max'):
            self.value=-1e9
        else:
            self.value=1e9
    def check_and_save(self,model:torch.nn.Module,value):
        save=False
        if(self.mode =='max'):
            if(value>self.value):
                if(self.verbose==1):
                    print(colorstr('blue', 'bold', f'\n model saved with value {value:.3f} previous is {self.value:.3f}'))
                self.value=value
                save=True
        if(self.mode == 'min'):
            if(value<self.value):
                if(self.verbose==1):
                    print(colorstr('blue', 'bold', f'\n model saved with value {value:.3f} previous is {self.value:.3f}'))
                self.value=value
                save=True
        if(save):
            torch.save(model.state_dict(),self.model_filename)
        return 


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
                
                



class MyTrain():
    def __init__(self, model, train_loader, val_loader, loss, 
                 optimizer, ckpt_callback, early_stopping, epochs, scheduler):
        self.MODEL = model
        self.TRAIN_LOADER = train_loader
        self.VAL_LOADER = val_loader
        self.LOSS = loss
        self.OPTIMIZER = optimizer
        self.CKPT_CALLBACK = ckpt_callback
        self.EPOCHS = epochs
        self.EARLY_STOPPING = early_stopping
        self.SCHEDULAR = scheduler
        self.BEST_RMSE = 100000.0
        
        
        
    def run(self):
        for e in range(self.EPOCHS):
            LOGGER.info(('%12s' * 5) % ('Epoch', 'gpu_mem', 'loss', 'RMSE', 'LR'))
            pbar=tqdm(self.TRAIN_LOADER, total=len(self.TRAIN_LOADER), position=0, leave=True)
            acc_loss = 0
            acc_RMSE = 0
            self.MODEL.train()
            for i, (img_batch, meta_batch, y) in enumerate(pbar):
                # data loading :
                img_batch   = img_batch.to(DEVICE, dtype=torch.float)
                meta_batch   = meta_batch.to(DEVICE, dtype=torch.float)
                y = y.to(DEVICE, dtype=torch.float)
                
                # predict = forward pass with our model
                y_predicted = self.MODEL(img_batch, meta_batch)
                # calculate loss 
                l = self.LOSS(y_predicted.flatten(), y)
                
                RMSE = petfinder_rmse(y_predicted.detach().cpu(), y.cpu())
                # accumulative RMSE
                acc_RMSE  = (acc_RMSE * i + RMSE.item()) / (i + 1)
                # accumlated loss 
                acc_loss = (acc_loss * i + l.item()) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%12s' * 2 + '%12.4g' * 3) % (f'{e+1}/{self.EPOCHS}', mem, acc_loss, acc_RMSE, 
                                                                    self.SCHEDULAR.get_last_lr()[0]))
                
                
                #l *= 2
                # calculate gradients = backward pass
                l.backward()
                # update weights
                self.OPTIMIZER.step()
                # zero the gradients after updating
                self.OPTIMIZER.zero_grad()
                # schedular step
                self.SCHEDULAR.step()
            
            # evaluation 
            self.MODEL.eval()
            acc_loss = 0
            acc_RMSE = 0
            with torch.no_grad():
                LOGGER.info(('%12s' * 5) % ('', 'gpu_mem', 'loss', 'RMSE', 'LR'))
                pbar=tqdm(self.VAL_LOADER, total=len(self.VAL_LOADER), position=0, leave=True)
        
                for j, (img_batch, meta_batch, y) in enumerate(pbar):
                    # data loading :
                        img_batch   = img_batch.to(DEVICE, dtype=torch.float)
                        meta_batch   = meta_batch.to(DEVICE, dtype=torch.float)
                        y = y.to(DEVICE, dtype=torch.float)
                        # predict = forward pass with our model
                        y_predicted = self.MODEL(img_batch, meta_batch)
                        l = self.LOSS(y_predicted.flatten(), y)
                        
                        RMSE = petfinder_rmse(y_predicted.detach().cpu(), y.cpu())
                        # accumulative RMSE
                        acc_RMSE  = (acc_RMSE * j + RMSE.item()) / (j + 1)
                        # accumlated loss 
                        acc_loss = (acc_loss * j + l.item()) / (j + 1)  # update mean losses
                        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                        pbar.set_description(('%12s' * 2 + '%12.4g' * 3) % ('     ',mem, acc_loss, acc_RMSE,
                                                                            self.SCHEDULAR.get_last_lr()[0]))
            # save model 
            self.CKPT_CALLBACK.check_and_save(self.MODEL, acc_RMSE)
            # save best rmse
            if acc_RMSE < self.BEST_RMSE:
                self.BEST_RMSE = acc_RMSE
            # early stopping     
            self.EARLY_STOPPING(acc_RMSE)
            if self.EARLY_STOPPING.early_stop:
                break
            
    
        
            
            



# get pawpularity value : 
def petfinder_rmse(input,target):
    return 100*torch.sqrt(F.mse_loss(torch.sigmoid(input.flatten()), target))


# get pawpularity value : 
#def petfinder_rmse(input,target):
#    return 100*torch.sqrt(F.mse_loss(input.flatten(), target))


class LargeSwinTransformerBackbone(nn.Module):
    def __init__(self, pretrained=True) -> None:
        super(LargeSwinTransformerBackbone,self).__init__()
        self.backbone=create_model(model_name='swin_large_patch4_window7_224',
                                   pretrained=pretrained,
                                   num_classes=0)
        # freeze some weights
        #l =[layer for layer in self.backbone.parameters()]
        #for layer in l[:300]:
        #   layer.requires_grad=False
        
    def forward(self,x):
        x=self.backbone(x)
        return x
    

class PawPularitySwin(nn.Module):
    def __init__(self, pretrained=True):
        super(PawPularitySwin, self).__init__()
        
        # backbone 
        self.backbone = LargeSwinTransformerBackbone(pretrained=pretrained)

        self.lin1 = nn.Linear(12, 64)
        self.drop1 = nn.Dropout(0.4)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(1568,1)
        
 
    def forward(self, x1, x2):
        # batch size       
        image_out = self.backbone(x1)  

        
        # meta data branch :
        x = torch.sigmoid(self.lin1(x2))
        x = self.drop1(x)
        meta_output = torch.sigmoid(self.lin2(x))
        # concatenate two outputs
        x = torch.cat((image_out, meta_output), dim=1)
        # head :
        x = self.lin3(x)
        return x










IMG_SIZE = (224, 224)
BATCH_SIZE = 2
EPOCHS=5
LR=4e-5
DEVICE = 'cuda:0'
NFOLDS = 5


df = pd.read_csv('train.csv')
img_path_train = 'train/'
df = df.sample(frac=1).reset_index(drop=True)
df['Id'] = df['Id'].apply(lambda x: img_path_train + x +'.jpg')
df['norm_score'] = df['Pawpularity']/100
num_bins = int(np.ceil(2*((len(df))**(1./3))))

df['bins'] = pd.cut(df['norm_score'], bins=num_bins, labels=False)
print(df.head())
#
weight_vec = list(df.groupby('bins').count()['Id'])
weight_vec = [1/i for i in weight_vec]
weight_vec_normalize = [i/sum(weight_vec) for i in weight_vec]
plt.plot(weight_vec_normalize)
plt.show()





skf = StratifiedKFold(n_splits=NFOLDS, random_state = seed, shuffle=True)
best_RMSE = []
for i, (train_df_index, val_df_index) in enumerate(skf.split(df.index, df['bins'])):
    if i!=4:
        print(f'Pass fold {i+1}')
        continue
    
    print(colorstr('blue', 'bold', f'****************** Fold {i+1} *********************'))
    train_df = df.iloc[train_df_index]
    val_df   = df.iloc[val_df_index]
    
    train_data = CustomDataGenerator(train_df, dim=IMG_SIZE, train_flag=True)
    val_data   = CustomDataGenerator(val_df,   dim=IMG_SIZE, train_flag=False)
    


    train_loader=DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)

    val_loader=DataLoader(dataset=val_data,
                         batch_size=BATCH_SIZE*2,
                         shuffle=False,
                         num_workers=4,
                         pin_memory=True)

    print(colorstr('blue', 'bold', 'DataLoading .....Done!'))
   
    model = PawPularitySwin()
    
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(DEVICE)



    print(colorstr('blue', 'bold', 'Creating Model .....Done!'))
    # summary(model.to(DEVICE), input_size=(3,*IMG_SIZE))


    #loss = CrossEntropyLoss(weight = torch.tensor(weight_vec_normalize).to(DEVICE))
    #loss = CrossEntropyLoss()
    loss = BCEWithLogitsLossFlat()
    #loss = MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    
    ckpt_callback = CheckpointCallback(f'swin_large_224_fold{i+1}.pt', 'min', verbose=1)
    early_stopping = EarlyStopping(patience=2)
    train_1 = MyTrain(model, 
                      train_loader, 
                      val_loader, 
                      loss, 
                      optimizer, 
                      ckpt_callback, 
                      early_stopping, 
                      EPOCHS, scheduler)
    
    train_1.run()
    best_RMSE.append(train_1.BEST_RMSE)
    
    
    # delete model :
    del train_1
    model.to('cpu')
    del model
    gc.collect()
    with torch.cuda.device(DEVICE): 
         torch.cuda.empty_cache()
         
         
    


   

