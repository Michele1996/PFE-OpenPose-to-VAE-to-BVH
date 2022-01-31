import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_data(name="keypoints_150_videos.npy",sequence = False):
    train_scenes = np.load(name,allow_pickle=True)
    size = 0
    x_train = []
    for scene in train_scenes :
        size += len(scene)
        for frame in scene :
            x_train.append(frame)
            
    print(len(train_scenes))
    x_train = np.array(x_train).reshape((len(train_scenes),50))
    x_train_x = x_train[:,::2]
    x_train_y = x_train[:,1::2]

    x_train = np.zeros((len(train_scenes),25,2))
    x_train[:,:,0] = x_train_x
    x_train[:,:,1] = x_train_y
        
    return x_train
def root_mean_squared_error_loss(y_true, y_pred):
     return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))


tab = load_data("keypoints_150_videos.npy")
train_scenes  = (tab)
max_seq = 0
for scene in train_scenes :
    if len(scene) > max_seq :
        max_seq = len(scene)
        
        


## Définition de l'architecture du modèle
model = keras.Sequential()
model.add(keras.layers.Input(shape=(100,50)))
model.add(keras.layers.Masking(mask_value=0.0))
model.add(keras.layers.LSTM(75, return_sequences=True, name='encoder_1'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(50, return_sequences=True, name='encoder_2'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(25, return_sequences=False, name='encoder_3'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.RepeatVector(100, name='encoder_decoder_bridge'))
model.add(keras.layers.LSTM(25,return_sequences=True, name='decoder_1'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(50,return_sequences=True, name='decoder_2'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(75,return_sequences=True, name='decoder_3'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(50)))
model.compile(optimizer="adam",loss=root_mean_squared_error_loss, metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])
model.build()
print(model.summary())
scaler = MinMaxScaler()
train_scenes=list(train_scenes)
# transform data
for i in range(len(train_scenes)):
    train_scenes[i]= scaler.fit_transform(train_scenes[i])

for i in range(len(train_scenes)) :
    #print(scene.shape)
    #print(len(scene),max_seq, type(scene))
    train_scenes[i]=train_scenes[i][:100]
train_scenes=train_scenes[:145100]
    
train_scenes= np.array(train_scenes).reshape(1451,100,50)
model.fit(train_scenes,train_scenes,batch_size=128, validation_split=0.2,epochs=500)
checkpoint_filepath = './checkpoint/AE_len_80_batch_128_nstep_weights.hdf5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath , verbose =1, monitor='loss', mode='min', save_best_only=True)