import sys 
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd 
import cv2
import random 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from albumentations import (Compose, Rotate, HorizontalFlip, RandomBrightnessContrast,
                            ChannelShuffle, VerticalFlip, RandomCrop, HueSaturationValue)



class MyDataGenerator(torch.utils.data.Dataset):
    
    def __init__(self, df, dim, train_flag = True):
        super(MyDataGenerator, self).__init__()
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
            HueSaturationValue()
            ])
        # define augmentation for validation 
        self.transform_val = Compose([
            ])
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.df)
    
       
    
    def __getitem__(self, idx):
        'Generates data containing batch_size samples'       
        # generate data :
        img = cv2.imread(self.df.iloc[idx, 0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.dim[:2])
        img = np.array(img, dtype='uint8')
        # data augmentation 
        img = self.__image_augmentation(img)
        img = img / 255.0 
        # transpose axis 
        img = img.transpose(2,1,0)
        
        meta = np.array(self.df.iloc[idx, 1:-1].values, dtype='float32')
        y = self.df.iloc[idx, -1]
            
        return torch.tensor(img), torch.tensor(meta), torch.tensor(y)
    
    def __image_augmentation(self, img):
        if self.train_flag:
            return self.transform_train(image=img)['image']
        else:
            return self.transform_val(image=img)['image']




        
if __name__ == '__main__':

    df = pd.read_csv('train.csv')
    img_path_train = 'train/'
    df['Id'] = df['Id'].apply(lambda x: img_path_train + x +'.jpg')
    train_df, val_df = train_test_split(df, 
                                        test_size=0.2, 
                                        random_state=999, 
                                        stratify=df["Pawpularity"])
    
    IMG_SIZE = (384, 384)
    BATCH_SIZE  = 8
    train_flow = MyDataGenerator(train_df, dim=IMG_SIZE, train_flag=True)
    val_flow  = MyDataGenerator(val_df, dim=IMG_SIZE,train_flag=False)
    
    train_loader=DataLoader(dataset=train_flow,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True)
    
    val_loader=DataLoader(dataset=val_flow,
                            batch_size=BATCH_SIZE*2,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)

