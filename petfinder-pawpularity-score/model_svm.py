import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import colorstr
from sklearn.svm import SVR
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
import pickle
from sklearn.metrics import mean_squared_error


tf.keras.backend.clear_session()
     
class BackBone(tf.keras.Model):
    def __init__(self, input_shape=(299,299,3), weights=None):
        super(BackBone, self).__init__()
        self.basemodel = tf.keras.applications.InceptionV3(input_shape = input_shape,
                                                           include_top=False, 
                                                           weights=weights)
        self.basemodel.trainable = True
        for layer in self.basemodel.layers[:-27]:
            layer.trainable = False
            
            
    def call(self, input_tensor):
        x = self.basemodel(input_tensor, training=False)
        return x
    
    # build graph
    def build_graph(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=[self.call(x)])
 
    
class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, pt_depth=2048):
        super(AttentionBlock, self).__init__()
        self.pt_depth = pt_depth
        self.drop1 = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Conv2D(64, kernel_size=(1,1), padding='same', activation='relu')
        self.dense2 = tf.keras.layers.Conv2D(16, kernel_size=(1,1), padding='same', activation='relu')
        self.dense3 = tf.keras.layers.Conv2D(8, kernel_size=(1,1), padding='same', activation='relu')
        self.dense4 = tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='valid', activation='sigmoid')
        self.upsample = tf.keras.layers.Conv2D(self.pt_depth,
                                              kernel_size=(1,1),
                                              padding='same',
                                              activation='linear',
                                              use_bias=False,
                                              weights=[np.ones((1,1,1,self.pt_depth))])
        self.upsample.trainable = False
        self.gap_m = tf.keras.layers.GlobalAveragePooling2D()
        self.gap_f = tf.keras.layers.GlobalAveragePooling2D()
        
        
        
    def call(self, input_tensor):
        x = self.drop1(input_tensor)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        attention_output = self.upsample(x)
        mask_features = tf.keras.layers.multiply([attention_output, input_tensor])
        gap_features = self.gap_f(mask_features)
        gap_mask     = self.gap_m(attention_output)
        gap = tf.keras.layers.Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
        return gap
        
    

class HeadModel(tf.keras.Model):
    def __init__(self):
        super(HeadModel, self).__init__()
        self.drop1 = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.drop2 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(1, activation='relu')
        
    def call(self, input_tensor):
        x = self.drop1(input_tensor)
        x = self.dense1(x)
        x = self.drop2(x)
        x = self.dense2(x)
        return x
    
    # build graph
    def build_graph(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=[self.call(x)])
    


class PawPularitywithMetadata(tf.keras.Model):
    def __init__(self, img_size, backbone_weights=None, with_head = True):
        super(PawPularitywithMetadata, self).__init__()
        self.img_size = img_size
        self.with_head = with_head
        
        # backbone 
        self.backbone1 = BackBone(input_shape=(*img_size, 3), 
                                  weights = backbone_weights)
        # neck
        # self.gap = tf.keras.layers.GlobalAveragePooling2D()
        # attention block
        self.atten = AttentionBlock(self.backbone1.layers[0].output_shape[-1])
        self.dense1 = tf.keras.layers.Dense(526, activation='relu')
        
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.drop1 = tf.keras.layers.Dropout(0.3)
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')
        
        self.concat = tf.keras.layers.Concatenate(name = 'ConCat_Layer')
        # head
        self.head = HeadModel()
        
        
    def call(self, input_tensor, train_=True):
        x = input_tensor[0]
        if train_:
            # x = image_augmentation(x)
            pass
        x = self.backbone1(x)
        
        x = self.atten(x)
        
        #x = self.gap(x)
        # image outputs
        image_out = self.dense1(x)
        
        # meta data branch :
        x = self.dense2(input_tensor[1])
        x = self.drop1(x)
        meta_output = self.dense3(x)
        x = self.concat([image_out, meta_output])
        # head :
        if self.with_head:
            x = self.head(x)
        return x
    
    # build graph
    def build_graph(self, image_shape, meta_shape):
        x1 = tf.keras.layers.Input(shape=image_shape)
        x2 = tf.keras.layers.Input(shape=meta_shape)
        return tf.keras.Model(inputs=[x1, x2], outputs=[self.call([x1, x2])])
        
        
IMG_SIZE = (299, 299)
BATCH_SIZE = 64       
# original model

pawpularitymodel = PawPularitywithMetadata(IMG_SIZE,  
                                           with_head=True)
# model summary
pawpularitymodel.build([(None, *IMG_SIZE, 3), (None,12)])
print(colorstr('green', 'bold', 'Creating Model... done!'))  
pawpularitymodel.build_graph((*IMG_SIZE, 3), 12).summary()
pawpularitymodel.load_weights('inception_attention.hdf5')
print(colorstr('green', 'bold', 'Model Build and weights loaded... done!')) 



SVR_model = PawPularitywithMetadata(IMG_SIZE,  
                                    with_head=True)
SVR_model.build([(None, *IMG_SIZE, 3), (None,12)])
SVR_model.load_weights('inception_attention.hdf5')
SVR_model.with_head =False
# model summary
print(colorstr('green', 'bold', 'SVR model created ... done!'))  
SVR_model.build_graph((*IMG_SIZE, 3), 12).summary()


df = pd.read_csv('train.csv')
img_path_train = 'train/'
# df['Id'] = df['Id'].apply(lambda x: img_path_train + x +'.jpg')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=999, stratify=df["Pawpularity"])



SVM_train_data = {'Embedding':[], 'Pawpularity':[]}
for index, row in tqdm(df.iterrows(), total=len(df)):
    image_path = 'train/'+row['Id'] + '.jpg'
    test_image = image.load_img(image_path, target_size=IMG_SIZE)
    test_image = image.img_to_array(test_image)
    test_image = test_image * 1/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    
    meta_data = np.array(row[1:-1], dtype='int')
    meta_data = np.expand_dims(meta_data, axis=0)
    pred = SVR_model.predict([test_image, meta_data])
    
    SVM_train_data['Embedding'].append(pred[0])
    SVM_train_data['Pawpularity'].append(row['Pawpularity'])
    
print(colorstr('green', 'bold', 'SVR train data created ... done!'))  
       
'''    
SVM_val_data = {'Embedding':[], 'Pawpularity':[]}
for index, row in tqdm(val_df.iterrows(), total=len(val_df)):
    image_path = 'train/'+row['Id'] + '.jpg'
    test_image = image.load_img(image_path, target_size=IMG_SIZE)
    test_image = image.img_to_array(test_image)
    test_image = test_image * 1/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    
    meta_data = np.array(row[1:-1], dtype='int')
    meta_data = np.expand_dims(meta_data, axis=0)
    pred = SVR_model.predict([test_image, meta_data])
    
    SVM_val_data['Embedding'].append(pred[0])
    SVM_val_data['Pawpularity'].append(row['Pawpularity'])    
    
    
print(colorstr('green', 'bold', 'SVR val data created ... done!'))     


SVM_all_data = deepcopy(SVM_train_data)
SVM_all_data['Embedding'] += SVM_val_data['Embedding']
SVM_all_data['Pawpularity'] += SVM_val_data['Pawpularity']
'''
regr = SVR(C=10, epsilon=0.2)
regr.fit(SVM_train_data['Embedding'], SVM_train_data['Pawpularity'])
print(colorstr('green', 'bold', 'SVR trained ... done!')) 
predictions = regr.predict(SVM_train_data['Embedding'])
error = np.sqrt(mean_squared_error(df['Pawpularity'], predictions))
    
    
                                                                    

#Save to file in the current working directory
pkl_filename = "SVRhead.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(regr, file)

