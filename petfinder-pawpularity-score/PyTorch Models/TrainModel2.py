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
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from albumentations import (Compose, Rotate, HorizontalFlip, RandomBrightnessContrast, Resize, 
                            ChannelShuffle, VerticalFlip, RandomCrop, HueSaturationValue)
from albumentations.augmentations.transforms import Blur, CoarseDropout
from fastai.vision.all import BCEWithLogitsLossFlat


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
            Rotate(limit=40, p=random.uniform(0.4, 0.6)),
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
            CoarseDropout(always_apply=False, p=0.5, max_holes=12, max_height=30, max_width=30, 
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
        
        # meta = np.array(self.df.iloc[idx, 1:-1].values, dtype='float32')
        y = self.df.iloc[idx, -2]
            
        return torch.tensor(img), torch.tensor(y)
    
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
            for i, (img_batch, y) in enumerate(pbar):
                # data loading :
                img_batch   = img_batch.to(DEVICE, dtype=torch.float)
                y = y.to(DEVICE, dtype=torch.float)
                
                # predict = forward pass with our model
                y_predicted = self.MODEL(img_batch)
                # calculate loss 
                l = self.LOSS(y_predicted, y)
                
                RMSE = petfinder_rmse(y_predicted, y)
                # accumulative RMSE
                acc_RMSE  = (acc_RMSE * i + RMSE.item()) / (i + 1)
                # accumlated loss 
                acc_loss = (acc_loss * i + l.item()) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%12s' * 2 + '%12.4g' * 3) % (f'{e}/{self.EPOCHS - 1}', mem, acc_loss, acc_RMSE, 
                                                                    self.SCHEDULAR.get_last_lr()[0]))
        
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
        
                for j, (img_batch, y) in enumerate(pbar):
                    # data loading :
                        img_batch   = img_batch.to(DEVICE, dtype=torch.float)
                        y = y.to(DEVICE, dtype=torch.float)
                        # predict = forward pass with our model
                        y_predicted = self.MODEL(img_batch)
                        l = self.LOSS(y_predicted, y)
                        
                        RMSE = petfinder_rmse(y_predicted, y)
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
            
            



# get pawpularity value : 
def petfinder_rmse(input,target):
    return 100*torch.sqrt(F.mse_loss(torch.sigmoid(input.flatten()), target))



IMG_SIZE = (224, 224)
BATCH_SIZE = 2
EPOCHS=2
LR=2e-5
DEVICE = 'cuda:1'
NFOLDS = 10


df = pd.read_csv('train.csv')
img_path_train = 'train/'
df['Id'] = df['Id'].apply(lambda x: img_path_train + x +'.jpg')
df['norm_score'] = df['Pawpularity']/100
num_bins = int(np.floor(1+np.log2(len(df))))


df['bins'] = pd.cut(df['norm_score'], bins=num_bins, labels=False)

skf = StratifiedKFold(n_splits=NFOLDS, random_state = 12, shuffle=True)
best_RMSE = []
for i, (train_df_index, val_df_index) in enumerate(skf.split(df.index, df['bins'])):
    
    print(colorstr('blue', 'bold', f'****************** Fold {i+1} *********************'))
    train_df = df.iloc[train_df_index]
    val_df   = df.iloc[val_df_index]
    
    train_data = CustomDataGenerator(train_df, dim=IMG_SIZE, train_flag=True)
    val_data   = CustomDataGenerator(val_df,   dim=IMG_SIZE, train_flag=False)
    


    train_loader=DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True)

    val_loader=DataLoader(dataset=val_data,
                         batch_size=BATCH_SIZE*2,
                         shuffle=False,
                         num_workers=8,
                         pin_memory=True)

    print(colorstr('blue', 'bold', 'DataLoading .....Done!'))
   
    model = create_model(model_name='swin_large_patch4_window7_224',
                         pretrained=True,
                         num_classes=1)
    
    #model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(DEVICE)



    print(colorstr('blue', 'bold', 'Creating Model .....Done!'))
    # summary(model.to(DEVICE), input_size=(3,*IMG_SIZE))


    #loss = CrossEntropyLoss()
    loss = BCEWithLogitsLossFlat()
    optimizer = Adam(model.parameters(), lr=LR)
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
    best_RMSE.append(train_1.BEST_RMSE.to('cpu').item())
    
    
    # delete model :
    del train_1
    model.to('cpu')
    del model
    gc.collect()
    with torch.cuda.device(DEVICE): 
         torch.cuda.empty_cache()
    


   

