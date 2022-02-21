import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


physical_devices_gpu = tf.config.list_physical_devices("GPU")[0]
physical_devices_cpu = tf.config.list_physical_devices("CPU")
tf.config.set_visible_devices([physical_devices_gpu] + physical_devices_cpu)
tf.config.experimental.set_memory_growth(physical_devices_gpu, True)

images_list = os.listdir("train")
print(len(images_list))

df = pd.read_csv("train.csv")
df["Id"] = df["Id"].apply(lambda x: x + ".jpg")
df["Pawpularity"] = df["Pawpularity"].apply(lambda x: x / 100)
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=999, stratify=df["Pawpularity"]
)

keys_id = train_df.keys()


tf.keras.backend.clear_session()


def PawPularityModel(
    input_shape, weights="resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
):
    backbone = ResNet50(include_top=False, weights=weights, input_shape=input_shape)
    #for layer in backbone.layers[:-10]:
    #    layer.trainable = False

    backbone.layers[0]._name = "Image_Input"
    model1_output = backbone.output
    model1_output = tf.keras.layers.AveragePooling2D(pool_size=(3, 3))(model1_output)
    model1_output = tf.keras.layers.Flatten()(model1_output)
    model1_output = tf.keras.layers.Dense(256, activation="relu")(model1_output)
    
    
    model2 = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(12)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(64, activation="relu"),
        ]
    )
    model2.layers[0]._name = "MetaData_Input"
    # ------#
    new_output = tf.keras.layers.Concatenate(axis=1)([model1_output, model2.output])
    new_output = tf.keras.layers.Dropout(rate=0.2)(new_output)
    new_output = tf.keras.layers.Dense(64, activation="relu")(new_output)
    new_output = tf.keras.layers.Dense(1, activation="relu")(new_output)
    # new_output = tf.keras.layers.Lambda(lambda x: x*100)(new_output)

    MyModel = tf.keras.Model(
        inputs=[backbone.input, model2.input], outputs=[new_output]
    )
    return MyModel


BATCH_SIZE = 12
INPUT_SHAPE = (224, 224, 3)

MyModel = PawPularityModel(input_shape=INPUT_SHAPE)
MyModel.load_weights('resnet50_saved_weights_1.hdf5')
MyModel.summary()

train_generator = ImageDataGenerator(
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1 / 255.0,
)

val_generator = ImageDataGenerator(rescale=1/255.0)


# ------------------------------ #
train_datagen_1 = train_generator.flow_from_dataframe(
    dataframe=train_df,
    directory="train",
    x_col="Id",
    y_col="Pawpularity",
    target_size=INPUT_SHAPE[:2],
    color_mode="rgb",
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

train_datagen_2 = train_generator.flow_from_dataframe(
    dataframe=train_df,
    directory="train",
    x_col="Id",
    y_col=list(keys_id[1:-1]),
    target_size=INPUT_SHAPE[:2],
    color_mode="rgb",
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False,
)
# ------------------------------ #
val_datagen_1 = val_generator.flow_from_dataframe(
    dataframe=val_df,
    directory="train",
    x_col="Id",
    y_col="Pawpularity",
    target_size=INPUT_SHAPE[:2],
    color_mode="rgb",
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

val_datagen_2 = val_generator.flow_from_dataframe(
    dataframe=val_df,
    directory="train",
    x_col="Id",
    y_col=list(keys_id[1:-1]),
    target_size=INPUT_SHAPE[:2],
    color_mode="rgb",
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False,
)


def gen_flow_for_2_inputs(datagen_1, datagen_2):
    while True:
        X1 = datagen_1.next()
        X2 = datagen_2.next()
        yield [X1[0], X2[1]], X1[1]
        
        
train_flow = gen_flow_for_2_inputs(train_datagen_1, train_datagen_2)
val_flow   = gen_flow_for_2_inputs(val_datagen_1, val_datagen_2)


model_callbacks = [
    ModelCheckpoint(
        filepath = 'resnet50_saved_weights_2.hdf5',
        save_weights_only = True,
        monitor = 'val_RMSE',
        mode = 'min',
        save_best_only = True),
    ReduceLROnPlateau(
        monitor="val_RMSE", 
        factor=0.8,
        patience=2, 
        verbose=1
    ),
]

MyModel.compile(optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01),
                loss=['mse'],
                metrics = tf.keras.metrics.RootMeanSquaredError(name='RMSE'))

hist = MyModel.fit(train_flow,
                   epochs=10,
                   validation_data=val_flow,
                   steps_per_epoch=len(train_df) // BATCH_SIZE,
                   validation_steps=len(val_df) // BATCH_SIZE,
                   callbacks=[model_callbacks])


