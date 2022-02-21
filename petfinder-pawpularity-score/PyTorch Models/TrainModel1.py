import sys 
import os
import math
from utils import colorstr, LOGGER
sys.path.append(os.path.abspath("pytorch-image-models"))
from timm import create_model
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from mydataloader import MyDataGenerator
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader




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
                
                
                

class LargeSwinTransformerBackbone(nn.Module):
    def __init__(self, pretrained=True) -> None:
        super(LargeSwinTransformerBackbone,self).__init__()
        self.backbone=create_model(model_name='swin_large_patch4_window12_384',
                                   pretrained=pretrained,
                                   num_classes=0)
        # freeze some weights
        #l =[layer for layer in self.backbone.parameters()]
        #for layer in l[:300]:
        #   layer.requires_grad=False
        
    def forward(self,x):
        x=self.backbone(x)
        return x
    

    
#model = LargeSwinTransformerBackbone().to('cuda:0')
#summary(model, input_size=(3,*IMG_SIZE))


class HeadModel(nn.Module):
    def __init__(self):
        super(HeadModel, self).__init__()
        self.drop1 = nn.Dropout(0.5)
        self.lin1  = nn.Linear(558, 128)
        self.drop2 = nn.Dropout(0.2)
        self.lin2  = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.drop1(x)
        x = F.relu(self.lin1(x))
        x = self.drop2(x)
        x = F.relu(self.lin2(x))
        return x
    
   
#head_model = HeadModel().to('cuda:0')
#summary(head_model, input_size=(558,))



class PawPularitySwin(nn.Module):
    def __init__(self, with_head = True, pretrained=True):
        super(PawPularitySwin, self).__init__()
        self.with_head = with_head
        
        # backbone 
        self.backbone = LargeSwinTransformerBackbone(pretrained=pretrained)

        self.lin1 = nn.Linear(1536, 526)
        self.lin2 = nn.Linear(12, 64)
        self.drop1 = nn.Dropout(0.3)
        self.lin3 = nn.Linear(64, 32)
        
        # head
        self.head = HeadModel()
        
        
    def forward(self, x1, x2):
        # batch size       
        x = self.backbone(x1)  
        # image outputs
        image_out = F.relu(self.lin1(x))
        
        # meta data branch :
        x = F.relu(self.lin2(x2))
        x = self.drop1(x)
        meta_output = F.relu(self.lin3(x))
        # concatenate two outputs
        x = torch.cat((image_out, meta_output), dim=1)
        # head :
        if self.with_head:
            x = self.head(x)
        return x


class MyTrain():
    def __init__(self, model, train_loader, val_loader, loss, optimizer, ckpt_callback, early_stopping, epochs):
        self.MODEL = model
        self.TRAIN_LOADER = train_loader
        self.VAL_LOADER = val_loader
        self.LOSS = loss
        self.OPTIMIZER = optimizer
        self.CKPT_CALLBACK = ckpt_callback
        self.EPOCHS = epochs
        self.EARLY_STOPPING = early_stopping
        
    def run(self):
        for e in range(self.EPOCHS):
            LOGGER.info(('%10s' * 4) % ('Epoch', 'gpu_mem', 'loss', 'RMSE'))
            pbar=tqdm(self.TRAIN_LOADER, total=len(self.TRAIN_LOADER), position=0, leave=True)
            train_mloss = 0
            self.MODEL.train()
            for i, (img_batch, meta_batch, y) in enumerate(pbar):
                # data loading :
                img_batch   = img_batch.to(DEVICE, dtype=torch.float)
                meta_batch  = meta_batch.to(DEVICE, dtype=torch.float)
                y = y.to(DEVICE, dtype=torch.float).view(img_batch.shape[0], 1)
                
                # predict = forward pass with our model
                y_predicted = self.MODEL(img_batch, meta_batch)
                # calculate loss 
                l = self.LOSS(y, y_predicted)
                # accumlated loss 
                train_mloss = (train_mloss * i + l) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 2) % (f'{e}/{self.EPOCHS - 1}', mem, train_mloss, math.sqrt(train_mloss)))
        
                # calculate gradients = backward pass
                l.backward()
                # update weights
                self.OPTIMIZER.step()
                # zero the gradients after updating
                self.OPTIMIZER.zero_grad()
            
            # evaluation 
            self.MODEL.eval()
            val_mloss = 0
            with torch.no_grad():
                LOGGER.info(('%10s' * 4) % ('', 'gpu_mem', 'loss', 'RMSE'))
                pbar=tqdm(self.VAL_LOADER, total=len(self.VAL_LOADER), position=0, leave=True)
        
                for j, (img_batch, meta_batch, y) in enumerate(pbar):
                    # data loading :
                        img_batch   = img_batch.to(DEVICE, dtype=torch.float)
                        meta_batch  = meta_batch.to(DEVICE, dtype=torch.float)
                        y = y.to(DEVICE, dtype=torch.float).view(img_batch.shape[0], 1)
                        # predict = forward pass with our model
                        y_predicted = self.MODEL(img_batch, meta_batch)
                        l = self.LOSS(y, y_predicted)
                        val_mloss = (val_mloss * j + l) / (j + 1)  # update mean losses
                        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                        pbar.set_description(('%10s' * 2 + '%10.4g' * 2) % ('     ',mem, val_mloss, math.sqrt(val_mloss)))
            # save model 
            self.CKPT_CALLBACK.check_and_save(self.MODEL, float(math.sqrt(val_mloss)))
            self.EARLY_STOPPING(val_mloss)
            if self.EARLY_STOPPING.early_stop:
                break
            if train_mloss < (val_mloss-1):
                break









IMG_SIZE = (384, 384)
BATCH_SIZE = 2
EPOCHS=50
LR=5e-6
DEVICE = 'cuda:1'
NFOLDS = 5
pawpularitymodel = PawPularitySwin()
# summary(pawpularitymodel, input_size=[(3,*IMG_SIZE), (12,)])

df = pd.read_csv('train.csv')
img_path_train = 'train/'
df['Id'] = df['Id'].apply(lambda x: img_path_train + x +'.jpg')
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True)
for i, (train_df_index, val_df_index) in enumerate(skf.split(df.index, df['Pawpularity'])):
    
    print(colorstr('blue', 'bold', f'****************** Fold {i+1} *********************'))
    train_df = df.iloc[train_df_index]
    val_df   = df.iloc[val_df_index]
    
    print(colorstr('blue', 'bold', 'Creating Model .....'))
    pawpularitymodel = PawPularitySwin()
    #summary(pawpularitymodel.to('cuda'), [(3, *IMG_SIZE), (12,)])
    print(colorstr('blue', 'bold', 'Creating Model .....Done!'))
    pawpularitymodel.to(DEVICE)
    
    train_data = MyDataGenerator(train_df, dim=IMG_SIZE, train_flag=True)
    val_data   = MyDataGenerator(val_df,   dim=IMG_SIZE, train_flag=False)
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
    loss=MSELoss()
    optimizer = Adam(pawpularitymodel.parameters(), lr=LR)
    ckpt_callback = CheckpointCallback(f'swin_large_384_fold{i+1}.pt', 'min', verbose=1)
    early_stopping = EarlyStopping(patience=7)
    
    train_1 = MyTrain(pawpularitymodel, train_loader, val_loader, loss, optimizer, ckpt_callback, early_stopping, EPOCHS)
    train_1.run()





#torch.save(pawpularitymodel.state_dict(), 'swin_large.pt')
#model1 = PawPularitySwin(pretrained=False).to('cuda:0')
#model1.load_state_dict(torch.load('swin_large.pt'))
        
        
    
    
    
    

