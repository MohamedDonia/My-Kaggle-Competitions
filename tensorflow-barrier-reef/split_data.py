#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:25:32 2022

@author: MohamedDonia
"""
import os
import shutil

NFolds=5
split_ratio = 0.2

images = os.listdir('./images')



for i in range(NFolds):
    validation = images[i*int(split_ratio*len(images)):(i+1)*int(split_ratio*len(images))]
    training = list(set(images) - set(validation))
    os.mkdir(f'./fold{i+1}')
    os.mkdir(f'./fold{i+1}/training')
    os.mkdir(f'./fold{i+1}/training/images')
    os.mkdir(f'./fold{i+1}/training/labels')
    os.mkdir(f'./fold{i+1}/validation')
    os.mkdir(f'./fold{i+1}/validation/images')
    os.mkdir(f'./fold{i+1}/validation/labels')
    
    for item in training:
        shutil.copy(f'./images/{item}', f'./fold{i+1}/training/images/{item}')
        label = item.replace('.jpg', '.txt')
        shutil.copy(f'./labels/{label}', f'./fold{i+1}/training/labels/{label}')
        
    for item in validation:
        shutil.copy(f'./images/{item}', f'./fold{i+1}/validation/images/{item}')
        label = item.replace('.jpg', '.txt')
        shutil.copy(f'./labels/{label}', f'./fold{i+1}/validation/labels/{label}')
        
    