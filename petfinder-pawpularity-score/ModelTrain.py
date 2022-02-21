import pandas as pd 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint



physical_devices_cpu = tf.config.list_physical_devices('CPU')
physical_devices_gpu = tf.config.list_physical_devices('GPU')[1]
tf.config.set_visible_devices(physical_devices_cpu + [physical_devices_gpu])
tf.config.experimental.set_memory_growth(physical_devices_gpu, True)






images_list = os.listdir('train')
print(len(images_list))

df = pd.read_csv('train.csv')
df['Id'] = df['Id'].apply(lambda x: x+'.jpg')
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

keys_id = train_df.keys()


tf.keras.backend.clear_session()

        
def PawPularityModel(input_shape, weights='efficientnetb2_notop.h5'):
    backbone = EfficientNetB2(
        include_top=False, 
        weights='efficientnetb2_notop.h5',
        input_shape = input_shape
        )
    backbone.layers[0]._name='Image_Input'
    model1_output = backbone.output
    model1_output = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(model1_output)
    model1_output = tf.keras.layers.Flatten()(model1_output)
    
    model2 = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(12)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(100, activation='relu')
    ]
    )
    model2.layers[0]._name = "MetaData_Input"
    #------#
    new_output = tf.keras.layers.Concatenate(axis=1)([model1_output, model2.output])
    new_output = tf.keras.layers.Dropout(rate=0.3)(new_output)
    new_output = tf.keras.layers.Dense(100, activation='relu')(new_output)
    new_output = tf.keras.layers.Dense(1, activation='relu')(new_output)
    
    MyModel = tf.keras.Model(inputs=[backbone.input, model2.input],
                         outputs=[new_output])
    return MyModel
    
    
    

BATCH_SIZE = 12
INPUT_SHAPE = (260, 260, 3)

MyModel = PawPularityModel(input_shape=INPUT_SHAPE)
MyModel.summary()




train_generator = ImageDataGenerator(
    rotation_range=0.2, 
    width_shift_range=0.2,
    height_shift_range=0.2, 
    brightness_range=[0.3,0.7], 
    shear_range=0.2, 
    zoom_range=0.3,
    channel_shift_range=0.3, 
    fill_mode='nearest',
    horizontal_flip=True, 
    vertical_flip=False, 
    rescale=1/255.0
)

val_generator = ImageDataGenerator(rescale=1/255.0)
        

# ------------------------------ #
train_datagen_1 = train_generator.flow_from_dataframe(
    dataframe=train_df, 
    directory='train', 
    x_col='Id', 
    y_col='Pawpularity',
    target_size=INPUT_SHAPE[:2], 
    color_mode='rgb',
    class_mode="raw", 
    batch_size=BATCH_SIZE,
    shuffle = False
)

train_datagen_2 = train_generator.flow_from_dataframe(
    dataframe=train_df, 
    directory='train', 
    x_col='Id', 
    y_col=list(keys_id[1:-1]),
    target_size=INPUT_SHAPE[:2], 
    color_mode='rgb',
    class_mode="raw", 
    batch_size=BATCH_SIZE,
    shuffle = False
)
# ------------------------------ #
val_datagen_1 = val_generator.flow_from_dataframe(
    dataframe=val_df, 
    directory='train', 
    x_col='Id', 
    y_col='Pawpularity',
    target_size=INPUT_SHAPE[:2], 
    color_mode='rgb',
    class_mode="raw", 
    batch_size=BATCH_SIZE,
    shuffle = False
)

val_datagen_2 = val_generator.flow_from_dataframe(
    dataframe=val_df, 
    directory='train', 
    x_col='Id', 
    y_col=list(keys_id[1:-1]),
    target_size=INPUT_SHAPE[:2], 
    color_mode='rgb',
    class_mode="raw", 
    batch_size=BATCH_SIZE,
    shuffle = False
)

def gen_flow_for_2_inputs(datagen_1, datagen_2):
    while True:
        X1 = datagen_1.next()
        X2 = datagen_2.next()
        yield [X1[0], X2[1]], X1[1]
        
        
train_flow = gen_flow_for_2_inputs(train_datagen_1, train_datagen_2)
val_flow   = gen_flow_for_2_inputs(val_datagen_1, val_datagen_2)


model_callbacks = ModelCheckpoint(
    filepath = 'saved_weights.hdf5',
    save_weights_only = True,
    monitor = 'val_RMSE',
    mode = 'max',
    save_best_only = True
)

MyModel.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss=['mse'],
                metrics = tf.keras.metrics.RootMeanSquaredError(name='RMSE'))

hist = MyModel.fit(train_flow,
                   epochs=10,
                   validation_data=val_flow,
                   steps_per_epoch=len(train_df) // BATCH_SIZE,
                   validation_steps=len(val_df) // BATCH_SIZE,
                   callbacks=[model_callbacks])
