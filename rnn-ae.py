import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def root_mean_squared_error_loss(y_true, y_pred):
     return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))


tab = np.load("kp_3.npy",allow_pickle=True)
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
train_scenes=list(train_scenes)
scaler = MinMaxScaler()
# transform data
for i in range(len(train_scenes)):
    train_scenes[i]= scaler.fit_transform(train_scenes[i])

for i in range(len(train_scenes)) :
    #print(scene.shape)
    #print(len(scene),max_seq, type(scene))
    zeros=np.zeros((max_seq-len(train_scenes[i]),50))
    train_scenes[i]=np.concatenate((train_scenes[i],zeros))
    train_scenes[i]=train_scenes[i][:100]
    
train_scenes= np.array(train_scenes).reshape(len(train_scenes),100,50)
model.fit(train_scenes,train_scenes,batch_size=128, validation_split=0.3,epochs=5000)
checkpoint_filepath = './checkpoint/AE_len_80_batch_128_nstep_weights.hdf5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath , verbose =1, monitor='loss', mode='min', save_best_only=True)