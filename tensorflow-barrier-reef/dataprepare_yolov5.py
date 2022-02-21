"""
Created on Tue Jan 11 15:04:52 2022

@author: Mohamed Donia
"""

# import liberaries:
import ast
import os
import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from shutil import copyfile
import warnings
warnings.filterwarnings("ignore")




def get_annotations(data):
    data  = ast.literal_eval(data)
    output_list = []
    for item in data:
        output_list.append([item['x'], item['y'], item['width'], item['height']])
        
    return output_list


def get_path(row):
    row['image_path'] = f'{train_path}/video_{row.video_id}/{row.video_frame}.jpg'
    return row

def Intersection(region, box):
    intersection = [max(region[0], box[0]),
                    max(region[1], box[1]), 
                    min(region[2], box[2]), 
                    min(region[3], box[3])]
    area = max(0, (intersection[2] - intersection[0])) * max(0, (intersection[3] - intersection[1]))
    if area <= 20*10:
        intersection = []
    return intersection

def PlotRect(img, bbox):
    for box in bbox:
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 225), 2)
    return img



def SplitImages(path, bbox, split_num=2, plot_flag=True):
    img = cv2.imread(path)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img_list = []
    for i in range(split_num):
        for j in range(split_num):
            # cut images 
            sub_img = img[int(i*h/split_num):int((i+1)*h/split_num), 
                          int(j*w/split_num):int((j+1)*w/split_num), :]
            img_list.append({'image_path':path,
                             'image':sub_img,
                             'boundary':[int(j*w/split_num), int(i*h/split_num),
                                     int((j+1)*w/split_num), int((i+1)*h/split_num)],
                             'bbox':[]})
            # plot sub images :
            if plot_flag:
                cv2.rectangle(img_RGB, 
                              (int(j*w/split_num), int(i*h/split_num)),
                              (int((j+1)*w/split_num), int((i+1)*h/split_num)),
                              (0, 0, 0), 2)
        
    for sub_image in img_list:
        for box in bbox:
            if len(box) != 0:
                box_ = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
                intersection = Intersection(sub_image['boundary'], box_)
                if len(intersection) !=0:
                    intersection = [intersection[0] - sub_image['boundary'][0],
                                    intersection[1] - sub_image['boundary'][1],
                                    intersection[2] - sub_image['boundary'][0],
                                    intersection[3] - sub_image['boundary'][1]]
                    sub_image['bbox'].append(intersection)
                
    img_list_with_bbox = [i for i in img_list ]#if len(i['bbox'])!=0]  
    #print(len(img_list_with_bbox), len(img_list_with_bbox[0]['bbox']))      
    
        
    if plot_flag:  
        for box in bbox:
            #print(box)
            box = [[box[0], box[1], box[0]+box[2], box[1]+box[3]]]
            img_RGB = PlotRect(img_RGB, box)
        plt.imshow(img_RGB)
        plt.show()
    return img_list_with_bbox
    

def YOLOV5Prepare(df):
    count = 0
    os.mkdir('dataset')
    os.mkdir('dataset/images')
    os.mkdir('dataset/labels')
    for index, item in tqdm(df.iterrows(), total=len(df)):
        img = cv2.imread(item['image_path'])
        h, w = img.shape[:2]
        bbox = item['bbox']
        cv2.imwrite(f'dataset/images/{count}.jpg', img)
        with open(f'dataset/labels/{count}.txt', 'w') as file:
            for box in bbox:
                box = [(box[0] + box[2]/2)/w, 
                       (box[1] + box[3]/2)/h, 
                       box[2]/w, 
                       box[3]/h]
                file.write(f'0 {box[0]} {box[1]} {box[2]} {box[3]} \n')
        count+=1


def train_test_split(path='dataset'):
    import random
    import shutil
    os.mkdir(path +  '/train')
    os.mkdir(path +  '/train/images')
    os.mkdir(path +  '/train/labels')
    os.mkdir(path +  '/validation')
    os.mkdir(path +  '/validation/images')
    os.mkdir(path +  '/validation/labels')
    images = os.listdir(path + '/images')
    
    num_img = len(images)
    random.shuffle(images)
    train_images = images[:int(0.8*num_img)]
    validation_images = images[int(0.8*num_img):]
    
    for img in train_images:
        shutil.move(path + '/images/'+img, path + '/train/images/'+img)
        shutil.move(path + '/labels/'+img.replace('.jpg', '.txt'), path + '/train/labels/'+img.replace('.jpg', '.txt'))
    for img in validation_images:
        shutil.move(path + '/images/'+img, path + '/validation/images/'+img)
        shutil.move(path + '/labels/'+img.replace('.jpg', '.txt'), path + '/validation/labels/'+img.replace('.jpg', '.txt'))
        
        
        

train_path = '/home/umbra/Work/Kaggle/tensorflow-barrier-reef/train_images'
df = pd.read_csv('train.csv')
print(df.head(10))
print(f'Number of rows: {len(df)}')
# Taken only annotated photos
df["num_bbox"] = df['annotations'].apply(lambda x: str.count(x, 'x'))
df_train = df #df[df["num_bbox"]>0]
print(f'Number of rows with annotations: {len(df_train)}')
#Annotations 
df_train['bbox'] = df_train['annotations'].apply(lambda x: get_annotations(x))
#Path of images
df_train = df_train.apply(get_path, axis=1)
df_train = df_train.reset_index(drop=True)




                
                
#YOLOV5Prepare(df_train)	
#train_test_split()    
    






data = []
for index, item in tqdm(df_train.iterrows(), total=len(df_train)):
    out = SplitImages(df_train.loc[index, 'image_path'], df_train.loc[index, 'bbox'], split_num=2, plot_flag=False)
    data.append(out)


os.mkdir('cots_basic_images')
os.mkdir('cots_basic_images/images')
os.mkdir('cots_basic_images/labels')

count = 0
for item in data:
    for i in item:
        img = i['image']
        h, w = img.shape[:2]
        bbox = i['bbox']
        cv2.imwrite(f'cots_basic_images/images/{count}.jpg', img)
        if bbox:
            with open(f'cots_basic_images/labels/{count}.txt', 'w') as file:
                for box in bbox:
                    box = [(box[0] + box[2])/2/w, 
                           (box[1] + box[3])/2/h, 
                           (box[2] - box[0])/w, 
                           (box[3] - box[1])/h]
                    file.write(f'0 {box[0]} {box[1]} {box[2]} {box[3]} \n')
                
        count+=1
        


