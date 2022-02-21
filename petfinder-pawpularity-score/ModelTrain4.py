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
from utils import colorstr


physical_devices_gpu = tf.config.list_physical_devices("GPU")[0]
physical_devices_cpu = tf.config.list_physical_devices("CPU")
tf.config.set_visible_devices([physical_devices_gpu] + physical_devices_cpu)
tf.config.experimental.set_memory_growth(physical_devices_gpu, True)

df = pd.read_csv('train.csv')
img_path_train = 'train/'
df['Id'] = df['Id'].apply(lambda x: img_path_train + x +'.jpg')

keys_id = df.keys()
print(colorstr('green', 'bold', 'Loading Data... done!'))



    
# data augmentation stage with horizontal flipping, rotation, zooms, etc....
image_augmentation = tf.keras.Sequential([
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomRotation(0.3),
    preprocessing.RandomZoom(0.1),
    preprocessing.RandomHeight(0.1),
    preprocessing.RandomWidth(0.1),
    preprocessing.Rescaling(1./255)], name='data_augmentation')





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
            x = image_augmentation(x)
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
backbone_weights = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pawpularitymodel = PawPularitywithMetadata(IMG_SIZE, 
                                           backbone_weights, 
                                           with_head=True)
# model summary
pawpularitymodel.build([(None, *IMG_SIZE, 3), (None,12)])
print(colorstr('green', 'bold', 'Creating Model... done!'))  
pawpularitymodel.build_graph((*IMG_SIZE, 3), 12).summary()



class PawPularityDataLoader():
    def __init__(self, df, BATCH_SIZE, is_labelled=True):
        self.df = df
        self.BATCH_SIZE = BATCH_SIZE
        self.is_labelled = is_labelled
        

    def __call__(self):
        return self.create_dataset_metadata()
    
    
    def img_read(self):
        def read_img(image_path):
            img = tf.io.read_file(image_path)
            img = tf.io.decode_jpeg(img, channels=3)
            img = tf.cast(img, tf.float32)
            img = tf.image.resize(img, IMG_SIZE)
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



train_df, val_df = train_test_split(df, test_size=0.2, random_state=999, stratify=df["Pawpularity"])
train_flow = PawPularityDataLoader(train_df, BATCH_SIZE)()
val_flow = PawPularityDataLoader(val_df, BATCH_SIZE)()


print(colorstr('green', 'bold', 'Creating DataSet... done!'))


train_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath = 'inception_attention.hdf5',
        save_weights_only = True,
        monitor = 'val_loss',
        mode = 'min',
        save_best_only = True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=2, verbose=1
    ),
]

pawpularitymodel.compile(
        loss = 'mse',
        optimizer = Adam(learning_rate=0.001),
        metrics = [tf.keras.metrics.RootMeanSquaredError()],
    )

# fitting Model
print(colorstr('green', 'bold', 'Training Model ... begin!'))
pawpularitymodel.fit(
        train_flow,
        epochs = 15,
        steps_per_epoch = len(train_flow),
        validation_data = val_flow,
        validation_steps = len(val_flow),
        callbacks = train_callbacks,
    )
print(colorstr('green', 'bold', 'Training Model ... done!'))