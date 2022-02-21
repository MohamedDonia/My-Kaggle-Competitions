#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:14:36 2022

@author: umbra
"""

import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt


images_path = 'cots_basic_images/images'



images_names = os.listdir(images_path)
for i in images_names:
    print(i)
    image_path = images_path + '/' + i
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h,w = image.shape[:2]
    
    with open(image_path.replace('.jpg', '.txt').replace('images','labels') ,'r') as f:
        lines = f.readlines()
        for line in lines:
            data = [' '.join(line.split(' ')[:-5])]
            data.extend( line.split(' ')[-5:-1])
            data[1:] = (float(data[i]) for i in range(1, len(data)))
            label, x1, y1, x2, y2 = data[0], \
                                        (data[1] - data[3]/2) * w, \
                                        (data[2] - data[4]/2) * h, \
                                        (data[1] + data[3]/2) * w, \
                                        (data[2] + data[4]/2) * h
                           
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            print(label)
    
    plt.imshow(image)
    plt.show()
    
    input_ = input('Enter key...')