import pandas as pd 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, Activation, MaxPool2D
from tensorflow.keras.layers import Add, Multiply, Lambda, Input, Dense, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

df = pd.read_csv('train.csv')
img_path_train = 'train/'
df['Id'] = df['Id'].apply(lambda x: img_path_train + x +'.jpg')
df["Pawpularity"] = df["Pawpularity"]
keys_id = df.keys()

class PawPularityDataLoader():
    def __init__(self, df, BATCH_SIZE, is_labelled=True):
        self.df = df
        self.BATCH_SIZE = BATCH_SIZE
        self.is_labelled = is_labelled
        

    def __call__(self):
        dataset = self.create_dataset_metadata()
        return dataset
    
    
    def img_read(self):
        def read_img(image_path):
            img = tf.io.read_file(image_path)
            img = tf.io.decode_jpeg(img, channels=3)
            img = tf.cast(img, tf.float32)
            img = tf.image.resize(img, (224,224))
            return img
        
        def can_be_readed(path, label):
            return read_img(path), label

        return can_be_readed if self.is_labelled else read_img
    
    
    def create_dataset_metadata(self):
        # function to convert images 
        image_read = self.img_read()
        # creating dataset of image path and pawpularity score
        if self.is_labelled:
            input_dataset = tf.data.Dataset.from_tensor_slices((self.df["Id"].values, 
                                                                self.df.drop(['Id', 'Pawpularity'], axis=1).values
                                                                ))
            output_dataset = tf.data.Dataset.from_tensor_slices((self.df["Pawpularity"].values))
            # converting images to tensors
            input_dataset = input_dataset.map(image_read, 
                                              num_parallel_calls=tf.data.AUTOTUNE)
            # creating final dataset
            dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
            # spliting in batches
            dataset = dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            return dataset
        else :
            input_dataset_1 = tf.data.Dataset.from_tensor_slices((self.df["Id"].values))
            input_dataset_2 = tf.data.Dataset.from_tensor_slices((self.df.drop('Id', axis=1).values))
            # converting images to tensors
            input_dataset_1 = input_dataset_1.map(image_read, num_parallel_calls=tf.data.AUTOTUNE)
            #dataset = tf.data.Dataset.from_tensor_slices((input_dataset_1, input_dataset_2))
            dataset = tf.data.Dataset.zip(((input_dataset_1, input_dataset_2),)).batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            return dataset


BATCH_SIZE =16
train_df, val_df = train_test_split(df, test_size=0.2, random_state=999, stratify=df["Pawpularity"])
train_flow = PawPularityDataLoader(train_df, BATCH_SIZE)()
val_flow = PawPularityDataLoader(val_df, BATCH_SIZE)()

def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):
    
    if output_channels is None:
        output_channels = input.get_shape()[-1]
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

    x = Add()([x, input])
    return x


def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):

    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1] 
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = MaxPool2D(padding='same')(input)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

            ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        ## upsampling
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        ## skip connections
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = UpSampling2D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)

    return output

def Attention_ResNet(input_, n_channels=32):
    x = Conv2D(n_channels, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 16x16

    x = residual_block(x, input_channels=32, output_channels=128)
    x = attention_block(x, encoder_depth=2)

    x = residual_block(x, input_channels=128, output_channels=256, stride=2)  # 8x8
    x = attention_block(x, encoder_depth=1)

    x = residual_block(x, input_channels=256, output_channels=512, stride=2)  # 4x4
    x = attention_block(x, encoder_depth=1)

    x = residual_block(x, input_channels=512, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)

    x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1))(x)  # 1x1
    output = Flatten()(x)
    
    return output

def build_meta_data_model(inputs):
    x = Dense(12, activation='relu')(inputs)
    for i in range(3):
        if i == 0: x = inputs
        x = Dense(12, activation='relu')(x)
        if (i + 1) % 2 == 0:
            x = BatchNormalization()(x)
            x = Concatenate()([x, inputs])
    return x

def get_model():
    image_inputs = Input((224, 224 , 3))
    meta_data_inputs = Input(12)
    
    image_x = Attention_ResNet(image_inputs)
    #image_x = Dense(1024, activation='relu')(image_x)
    #image_x = Dense(512, activation='relu')(image_x)
    meta_data_x = build_meta_data_model(meta_data_inputs)
    
    x = Concatenate(axis=1)([image_x, meta_data_x])
    output = Dense(1)(x)
    model = Model(inputs=[image_inputs, meta_data_inputs], outputs=[output])
    return model

model = get_model()

callback_1 = ModelCheckpoint(
    filepath = './model_weights_ft_best_t.h5',
    save_weights_only = True,
    monitor = 'RMSE',
    mode = 'min',
    save_best_only = True
)

callback_2 = ModelCheckpoint(
    filepath = './model_weights_ft_best_v.h5',
    save_weights_only = True,
    monitor = 'val_RMSE',
    mode = 'min',
    save_best_only = True
)

model.compile(optimizer = 'Adadelta',
                loss=['mse'],
                metrics = tf.keras.metrics.RootMeanSquaredError(name='RMSE'))

History = model.fit(train_flow,
                    validation_data= val_flow,
                    epochs=100,
                    steps_per_epoch=len(train_flow),
                    validation_steps=len(val_flow),
                    callbacks=[callback_1, callback_2])
    
model.save_weights('./model_weights_ft_last.h5')

