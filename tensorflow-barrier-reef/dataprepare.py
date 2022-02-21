"""
Created on Tue Jan 11 15:04:52 2022

@author: Mohamed Donia
"""

# import liberaries:
import ast
import os
import json
import pandas as pd
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


train_path = '/home/umbra/Work/Kaggle/tensorflow-barrier-reef/train_images'
def get_path(row):
    row['image_path'] = f'{train_path}/video_{row.video_id}/{row.video_frame}.jpg'
    return row



annotion_id = 0
def dataset2coco(df):
    
    global annotion_id
    
    annotations_json = {
        "info": [],
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    
    # info :
    info = {
        "year": "2022",
        "version": "1",
        "description": "COTS dataset - COCO format",
        "contributor": "",
        "url": "https://kaggle.com",
        "date_created": "2022-1-11T17:00:00"
    }
    annotations_json["info"].append(info)
    
    # licence :
    lic = {
            "id": 1,
            "url": "",
            "name": "Unknown"
        }
    annotations_json["licenses"].append(lic)
    
    # categories :
    classes = {"id": 0, "name": "starfish", "supercategory": "none"}

    annotations_json["categories"].append(classes)

   
    for ann_row in df.itertuples():
        # images :   
        images = {
            "id": ann_row[0],
            "license": 1,
            "file_name": ann_row.image_id + '.jpg',
            "height": ann_row.height,
            "width": ann_row.width,
            "date_captured": "2022-1-11T17:00:00"
        }
        
        annotations_json["images"].append(images)
        
        # annotations :
        bbox_list = ann_row.bbox
        
        for bbox in bbox_list:
            b_width = bbox[2]
            b_height = bbox[3]
            
            # some boxes in COTS are outside the image height and width
            if (bbox[0] + bbox[2] > 1280):
                b_width = bbox[0] - 1280 
            if (bbox[1] + bbox[3] > 720):
                b_height = bbox[1] - 720 
                
            image_annotations = {
                "id": annotion_id,
                "image_id": ann_row[0],
                "category_id": 0,
                "bbox": [bbox[0], bbox[1], b_width, b_height],
                "area": bbox[2] * bbox[3],
                "segmentation": [],
                "iscrowd": 0
            }
            
            annotion_id += 1
            annotations_json["annotations"].append(image_annotations)
           
    print(f"Dataset COTS annotation to COCO json format completed! Files: {len(df)}")
    return annotations_json


def save_annot_json(json_annotation, filename):
    with open(filename, 'w') as f:
        output_json = json.dumps(json_annotation)
        f.write(output_json)






df = pd.read_csv('train.csv')
print(df.head(10))
print(f'Number of rows: {len(df)}')

# Taken only annotated photos
df["num_bbox"] = df['annotations'].apply(lambda x: str.count(x, 'x'))
df_train = df[df["num_bbox"]>0]
print(f'Number of rows with annotations: {len(df_train)}')


#Annotations 
df_train['bbox'] = df_train['annotations'].apply(lambda x: get_annotations(x))

#Images resolution
df_train["width"] = 1280
df_train["height"] = 720

#Path of images
df_train = df_train.apply(get_path, axis=1)
df_train = df_train.reset_index(drop=True)

train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42)

train_annot_json = dataset2coco(train_df)
val_annot_json = dataset2coco(val_df)



os.mkdir('dataset')
os.mkdir('dataset/images')
os.mkdir('dataset/images/train2017')
os.mkdir('dataset/images/val2017')
os.mkdir('dataset/images/annotations')


# copy images :
for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
    copyfile(f'{row.image_path}', f'dataset/images/train2017/{row.image_id}.jpg')
    
for index, row in tqdm(val_df.iterrows(), total=len(val_df)):
    copyfile(f'{row.image_path}', f'dataset/images/val2017/{row.image_id}.jpg')    



# Save converted annotations
save_annot_json(train_annot_json, "dataset/images/annotations/train.json")
save_annot_json(val_annot_json, "dataset/images/annotations/valid.json")