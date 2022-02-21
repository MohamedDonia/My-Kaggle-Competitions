import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.model_selection import train_test_split


df = pd.read_csv('train.csv')

img_path_train = 'train/'
df['Id'] = df['Id'].apply(lambda x: img_path_train + x +'.jpg')


IMG_SIZE = (229, 229)
BATCH_SIZE = 64


def img_read(is_labelled):
    def read_img(image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, IMG_SIZE)
        return img
    
    def can_be_readed(path, label):
        return read_img(path), label

    return can_be_readed if is_labelled else read_img



def creat_dataset_metadata(df, batch_size, is_labelled = True):
    # function to convert images 
    image_read = img_read(is_labelled)
    
    # creating dataset of image path and pawpularity score
    if is_labelled:
        input_dataset = tf.data.Dataset.from_tensor_slices((df["Id"].values, df.drop(['Id', 'Pawpularity'], axis=1).values))
        output_dataset = tf.data.Dataset.from_tensor_slices((df["Pawpularity"].values))
        
        # converting images to tensors
        input_dataset = input_dataset.map(image_read, num_parallel_calls=tf.data.AUTOTUNE)
        
        # creating final dataset
        dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
        
        # spliting in batches
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
        
    else :
        input_dataset_1 = tf.data.Dataset.from_tensor_slices((df["Id"].values))
        input_dataset_2 = tf.data.Dataset.from_tensor_slices((df.drop('Id', axis=1).values))
        
        # converting images to tensors
        input_dataset_1 = input_dataset_1.map(image_read, num_parallel_calls=tf.data.AUTOTUNE)
        
#         dataset = tf.data.Dataset.from_tensor_slices((input_dataset_1, input_dataset_2))
        dataset = tf.data.Dataset.zip(((input_dataset_1, input_dataset_2),)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
def creat_dataset_image(df, batch_size, is_labelled = True):
    # function to convert images 
    image_read = img_read(is_labelled)
    
    # creating dataset of image path and pawpularity score
    if is_labelled:
        input_dataset = tf.data.Dataset.from_tensor_slices((df["Id"].values, df["Pawpularity"].values))
        
        # converting images to tensors
        dataset = input_dataset.map(image_read, num_parallel_calls=tf.data.AUTOTUNE)
        
        # spliting in batches
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
        
    else :
        input_dataset_1 = tf.data.Dataset.from_tensor_slices((df["Id"].values))
        
        # converting images to tensors
        input_dataset_1 = input_dataset_1.map(image_read, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = input_dataset_1.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset



# data augmentation stage with horizontal flipping, rotation, zooms, etc....
image_augmentation = tf.keras.Sequential([
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomRotation(0.3),
    preprocessing.RandomZoom(0.3),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    preprocessing.Rescaling(1./255)], name='data_augmentation')




inception_v3 = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = tf.keras.applications.InceptionV3(include_top=False, weights=inception_v3)
base_model.trainable = True

for layer in base_model.layers[:-28]:
    layer.trainable = False
    
def create_model_1():
    # image input model
    img_input = tf.keras.layers.Input(shape=(229, 229, 3), name='image_input')
    x = image_augmentation(img_input)
    x = base_model(x, training=False)
    x = GlobalMaxPooling2D()(x)
    x = layers.Dense(526, activation='relu')(x)
    output_layer = layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs = img_input, outputs=output_layer)
    
    return model


def create_model_2():
    # image input model
    img_input = tf.keras.layers.Input(shape=(229, 229, 3), name='image_input')
    x = image_augmentation(img_input)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    img_output = layers.Dense(526, activation='relu')(x)
    img_model = tf.keras.Model(img_input, img_output)
 
    # other data Model
    data_input = layers.Input(shape=(12), name='data_input')
    x = layers.Dense(64, activation='relu')(data_input)
    x = layers.Dropout(0.3)(x)
    data_output = layers.Dense(32, activation='relu')(x)
    data_model = tf.keras.Model(data_input, data_output)

    # concatinating Model layers
    concat_layer = layers.Concatenate(name = 'concat_layer')([img_model.output, data_model.output])

    combined_dropout = layers.Dropout(0.5)(concat_layer)
    combined_dence = layers.Dense(128, activation='relu')(combined_dropout)
    final_dropout = layers.Dropout(0.2)(combined_dence)

    output_layer = layers.Dense(1, activation='relu')(final_dropout)

    model = tf.keras.Model(inputs = [img_model.input, data_model.input], outputs=output_layer)
    
    return model


model = create_model_2()
model.summary()


train_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath = 'inception_saved_weights_2.hdf5',
        save_weights_only = True,
        monitor = 'val_loss',
        mode = 'min',
        save_best_only = True),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=2, verbose=1
    ),
]

final_results = []

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=999, stratify=df["Pawpularity"]
)

train_dataset = creat_dataset_metadata(train_df, BATCH_SIZE, is_labelled=True)
val_dataset = creat_dataset_metadata(val_df, BATCH_SIZE, is_labelled=True)

model.compile(
        loss = 'mse',
        optimizer = Adam(learning_rate=0.002),
        metrics = [tf.keras.metrics.RootMeanSquaredError()],
    )

# fitting Model
print('model training \n')
model.fit(
        train_dataset,
        epochs = 15,
        steps_per_epoch = len(train_dataset),
        validation_data = val_dataset,
        validation_steps = len(val_dataset),
        callbacks = train_callbacks,
    )