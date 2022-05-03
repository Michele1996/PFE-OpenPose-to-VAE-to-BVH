import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.manifold import TSNE
from tensorflow import keras
import json
import os
import argparse
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

#skeleton BODY25
L = [[17,15],[15,0],[18,16],[16,0],[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,24],[11,22],[22,23],[8,12],[12,13],[13,14],[14,21],[14,19],[19,20]]



def load_data(name="kpoutput_test_3.npy",sequence = False):
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
    
###################### OUTILS FUNCTIONS seen @Doriand & @Emilien ###################################
def simple_visu(x_train,index):
    test_image=x_train[index].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
    print(test_image)
    encoded_img1=autoencoder.predict(test_image)
    print(encoded_img1)
    visu_skel(encoded_img1,0)
    
def sim_pca(data,components=3):
    pca = PCA(n_components=components)
    x = pca.fit_transform(data)
    #print(pca.explained_variance_ratio_)
    return(pca,x)

def interpolate_points(p1, p2, n=8):
	ratios = np.linspace(0, 1, num=n)
	vectors = list()
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
		vectors.append(v)
	return np.asarray(vectors)
    
def visu_skel(x_train,index):
    frame = x_train[index]
    plt.figure()
    plt.xlim(left = -1.1,right=1.1)
    plt.ylim(top=1.1,bottom=-1.1)
    plt.scatter(frame[:,0],-frame[:,1])
    
    for k in L:
        plt.plot([frame[k[0],0],frame[k[1],0]],[-frame[k[0],1],-frame[k[1],1]])
    plt.show()

    
def visu_interpo(x_train,nb_iter,nb_frames,test,scaler,latent_size=2,show=True):
    num_files=0
    np.random.seed(42)
    for o in range (nb_iter):
        #print(np.random.randint(len(x_train)))
        #print(np.random.randint(len(x_test)))
        test_image1=x_train[np.random.randint(len(x_train))].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
        test_image2=x_test[np.random.randint(len(x_test))].reshape(1,x_test[0].shape[0],x_test[0].shape[1])
        encoded_img1=encoder.predict(test_image1)
        encoded_img2=encoder.predict(test_image2)
        #print(decoder.predict(encoded_img1)[0])
        #print(scaler.inverse_transform(decoder.predict(encoded_img1)[0]))
        #print(scaler.inverse_transform(decoder.predict(encoded_img2)[0]))
        interpolated_images=interpolate_points(encoded_img1.flatten(),encoded_img2.flatten())
        interpolated_orig_images=interpolate_points(test_image1.flatten(),test_image2.flatten())
        predict = (encoder.predict(x_train[::50]))
        if(show):
            if latent_size > 2 :
                pca,x = sim_pca(predict)
                inter = pca.transform(interpolated_images)
                fig = plt.figure()
                fig.add_subplot(projection='3d')
                plt.scatter(x[:,0],x[:,1],x[:,2],label="X_train")
                plt.scatter(inter[:,0],inter[:,1],inter[:,2],label="Interpolation")
                plt.scatter(inter[0,0],inter[0,1],inter[0,2],label="Beginning")
                plt.scatter(inter[-1,0],inter[-1,1],inter[-1,2],label="End")
                plt.legend()
                plt.show()

            else : 
                plt.figure()
                plt.scatter(predict[:,0],predict[:,1],label="X_train")
                plt.scatter(interpolated_images[:,0],interpolated_images[:,1],label="Interpolation")
                plt.scatter(interpolated_images[0,0],interpolated_images[0,1],label="Beginning")
                plt.scatter(interpolated_images[-1,0],interpolated_images[-1,1],label="End")
                plt.legend()
                plt.show()
        interpolated_images.shape
        num_images = nb_frames
        for i, image_idx in enumerate(interpolated_images):
            
            inter = interpolated_images[i].reshape(1,interpolated_images[i].shape[0])
            frame = decoder.predict(inter)
            #print(scaler.inverse_transform(frame.reshape(25,2)))
            save_json_for_MocapNET(scaler.inverse_transform(frame.reshape(25,2)),num_files,test)
            num_files+=1
            if(show and num_images<=10):
                plt.figure(figsize=(20, 8))
                ax = plt.subplot(5, num_images,num_images+ i + 1)
                plt.scatter(frame[:,0],-frame[:,1],s=10)
                for k in L:
                    plt.plot([frame[k[0],0],frame[k[1],0]],[-frame[k[0],1],-frame[k[1],1]])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_xlim(left = -1.1,right=1.1)
                ax.set_ylim(top=1.1,bottom=-1.1)
                ax.set_title("Latent: {}".format(i))
                ax = plt.subplot(5, num_images,2*num_images+ i + 1)
                plt.scatter(frame[0::2],-frame[1::2],s=10)
                for k in L:
                    plt.plot([frame[2*k[0]],frame[2*k[1]]],[-frame[2*k[0]+1],-frame[2*k[1]+1]])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_xlim(left = -1.1,right=1.1)
                ax.set_ylim(top=1.1,bottom=-1.1)
                ax.set_title("Image: {}".format(i))
                plt.show()

def test_points(x_train,random=True,n=1,step=50):
    data = (encoder.predict(x_train[::step]))
    pca,x = sim_pca(data)
    fig = plt.figure()
    fig.add_subplot(projection="3d")
    plt.scatter(x[:,0],x[:,1],x[:,2],label="X_train")
    plt.legend()
    plt.show()

    for k in range(n):
        if random :
            point_1 = (np.max(x[:,0])- np.min(x[:,0])) * np.random.random() - np.min(x[:,0])
            point_2 = (np.max(x[:,1])- np.min(x[:,1])) * np.random.random() - np.min(x[:,1])
            point_3 = (np.max(x[:,2])- np.min(x[:,2])) * np.random.random() - np.min(x[:,2])
        else :
            print("x : ")
            point_1 = input()
            print("y : ")
            point_2 = input()
            print("z : ")
            point_3 = input()

        point = pca.inverse_transform([float(point_1),float(point_2),float(point_3)])
        constr = decoder.predict(np.array([point]))
        visu_skel(constr,0)

def latent_representation_tSNE(x_train,show = False):
    predict = (encoder.predict(x_train))
    tsne = TSNE(n_components=3, learning_rate='auto',init='random')
    X_embedded = tsne.fit_transform(predict)
    if show :
        plt.figure()
        plt.scatter(X_embedded[:,0],X_embedded[:,1])
        plt.show()
    return X_embedded, tsne

def create_gif(name = 'mygif.gif'):

    pos = [np.random.randint(0,len(x_train)) for k in range(10)]
    i=0
    files = []
    for k in range(0,len(pos)-1) :
        test_image1=x_train[pos[k]].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
        test_image2=x_train[pos[k+1]].reshape(1,x_train[0]. shape[0],x_train[0].shape[1])
        encoded_img1=encoder.predict(test_image1)
        encoded_img2=encoder.predict(test_image2)
        interpolated_images=interpolate_points(encoded_img1.flatten(),encoded_img2.flatten(),n=30)

        inter = interpolated_images.reshape(30,interpolated_images[:].shape[1])
        frames = (decoder.predict(inter))

        filenames = []
        
        for frame in frames:
            i+=1
            plt.scatter(frame[:,0],-frame[:,1],s=10)
            for k in L:
                plt.plot([frame[k[0],0],frame[k[1],0]],[-frame[k[0],1],-frame[k[1],1]])
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            
            # create file name and append it to a list
            filename = f'{i}.png'
            filenames.append(filename)
            files.append(filename)
            # save frame
            plt.savefig(filename)
            plt.close()
    # build gif
    with imageio.get_writer(name, mode='I') as writer:
        for filename in files:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(files):
        os.remove(filename)
        
###################### OUTILS FUNCTIONS ###################################

def save_json_for_MocapNET(frame,index,test=False):
    vector=[]
    print("PROCESS FRAME: ",index)
    for i in range(len(frame)):
           vector.append(round(float(frame[i][0]),2))
           vector.append(round(float(frame[i][1]),2))
           vector.append(float(0.80))
    vector=vector[:len(vector)-1]
    dicto={"pose_keypoints_2d":vector}
    print("VETTORE ",vector)
    data_set = {"version": 1.3, "people": [{"person_id":[-1],"pose_keypoints_2d":vector,"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}
    index="0000"+str(index)
    decalage=len(str(index))-5
    index=index[decalage:len(str(index))]
    path = os.getcwd()
    filename="colorFrame_0_"+str(index)+"_keypoints.json"
    if os.path.exists(path+"\\MocapNET-master\\OUTPUT_to_BVH\\"+filename):
          os.remove(path+"\\MocapNET-master\\OUTPUT_to_BVH\\"+filename)
    with open(filename, 'w') as fp:
        json.dump(data_set, fp,separators=(',', ':'))
        fp.close()
    if(test):
        if(not os.path.isdir(path+"\\Test")):
            os.mkdir(path+"\\Test")
        if(not os.path.isdir(path+"\\Test\\OUTPUT_to_BVH_PATS2")):
            os.mkdir(path+"\\Test\\OUTPUT_to_BVH_PATS2")
        if(os.path.isfile(path+"\\Test\\OUTPUT_to_BVH_PATS2\\"+filename)):
            os.remove(path+"\\Test\\OUTPUT_to_BVH_PATS2\\"+filename)
        os.rename(os.path.join(path, filename), os.path.join(path+"\\Test\\OUTPUT_to_BVH_PATS2",filename))
    else:
        if(not os.path.isdir(path+"\\MocapNET-master")):
           print("Error, MocapNET not found. Please install it to continue")
           quit()
        if(not os.path.isdir(path+"\\MocapNET-master\\OUTPUT_to_BVH")):
            os.mkdir(path+"\\MocapNET-master\\OUTPUT_to_BVH")
        os.rename(os.path.join(path, filename), os.path.join(path+"\\MocapNET-master\\OUTPUT_to_BVH",filename))
        
def root_mean_squared_error_loss(y_true, y_pred):
     return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true)))

def VAE(latent_size=2):
    ## Définition de l'architecture du modèle
    encoder_1_size = 256
    encoder_1_size_1 = 128
    encoder_1_size_2 = 64
    encoder_1_size_3 = 32
    latent_size = latent_size
    input_layer = tf.keras.layers.Input(shape = (25,2))
    flattened = tf.keras.layers.Flatten()(input_layer)
    encoder_1 = tf.keras.layers.Dense(encoder_1_size, activation = tf.keras.activations.linear)(flattened)
    encoder_1 = tf.keras.layers.BatchNormalization()(encoder_1)
    encoder_1 = tf.keras.layers.Dropout(0.2)(encoder_1)
    encoder_2 = tf.keras.layers.Dense(encoder_1_size_1,activation = 'relu',kernel_initializer='glorot_normal')(encoder_1)
    encoder_2 = tf.keras.layers.BatchNormalization()(encoder_2)
    encoder_2 = tf.keras.layers.Dropout(0.2)(encoder_2)
    encoder_3 = tf.keras.layers.Dense(encoder_1_size_2,activation = 'relu',kernel_initializer='glorot_normal')(encoder_2)
    encoder_3 = tf.keras.layers.BatchNormalization()(encoder_3)
    encoder_3 = tf.keras.layers.Dropout(0.2)(encoder_3)
    encoder_4 = tf.keras.layers.Dense(encoder_1_size_3,activation = 'relu',kernel_initializer='glorot_normal')(encoder_3)
    encoder_4 = tf.keras.layers.BatchNormalization()(encoder_4)
    encoder_4 = tf.keras.layers.Dropout(0.2)(encoder_4)
    latent = tf.keras.layers.Dense(latent_size)(encoder_4)
    encoder = tf.keras.Model(inputs = input_layer, outputs = latent, name = 'encoder')
    #encoder.summary()
    input_layer_decoder = tf.keras.layers.Input(shape = encoder.output.shape[1:])
    decoder_1 = tf.keras.layers.Dense(encoder_1_size_3, activation = 'sigmoid',kernel_initializer='glorot_normal')(input_layer_decoder)
    decoder_1 = tf.keras.layers.BatchNormalization()(decoder_1)
    decoder_1 = tf.keras.layers.Dropout(0.2)(decoder_1)
    decoder_2 = tf.keras.layers.Dense(encoder_1_size_2, activation = 'sigmoid',kernel_initializer='glorot_normal')(decoder_1)
    decoder_2 = tf.keras.layers.BatchNormalization()(decoder_2)
    decoder_2 = tf.keras.layers.Dropout(0.2)(decoder_2)
    decoder_3 = tf.keras.layers.Dense(encoder_1_size_1, activation = 'sigmoid',kernel_initializer='glorot_normal')(decoder_2)
    decoder_3 = tf.keras.layers.BatchNormalization()(decoder_3)
    decoder_3 = tf.keras.layers.Dropout(0.2)(decoder_3)
    decoder_4 = tf.keras.layers.Dense(encoder_1_size, activation = 'sigmoid',kernel_initializer='glorot_normal')(decoder_3)
    decoder_4 = tf.keras.layers.BatchNormalization()(decoder_4)
    decoder_4 = tf.keras.layers.Dropout(0.2)(decoder_4)
    decoder_1 = tf.keras.layers.Dense(encoder.layers[1].output_shape[-1], activation = tf.keras.activations.linear)(decoder_4)
    constructed = tf.keras.layers.Reshape(x_train.shape[1:])(decoder_1)
    decoder = tf.keras.Model(inputs = input_layer_decoder, outputs = constructed, name= 'decoder')
    #decoder.summary()

    autoencoder = tf.keras.Model(inputs = encoder.input, outputs = decoder(encoder.output))
    autoencoder.summary()
    autoencoder.compile(optimizer="adam",loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return autoencoder,encoder,decoder

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("nb_iter", help="number of iter creation of nb_frames frames by encoder",type=int)
    parser.add_argument("nb_frames", help="number of frame create by encoder",type=int)
    parser.add_argument("test", help="path to data",type=bool,nargs='?',default=False)
    parser.add_argument("data_file_location", help="path to data",type=str,nargs='?',default="kpoutput_test_pats2_test_increasefps-test.npy")
    args = parser.parse_args()
    nb_frames=args.nb_frames
    nb_iter=args.nb_iter
    save_test=args.test
    x_train=load_data(name =args.data_file_location)
    autoencoder,encoder,decoder = VAE(latent_size = 5)
    scaler = MinMaxScaler()
    # transform data
    for i in range(len(x_train)):
        x_train[i]= scaler.fit_transform(x_train[i])
    print(len(x_train))
    #x_train, x_test = train_test_split(x_train,test_size=0.1, random_state=42)
    
    x_test=x_train[0:3000]
    x_train=x_train[3001:]
    history=autoencoder.fit(x_train,x_train,batch_size=64,validation_split=0.2,shuffle=True,epochs=100)
    
    #plt.figure()
    #plt.plot(history.history['loss'], label='loss')
    #plt.plot(history.history['val_loss'], label='val_loss')
    #plt.legend()
    #plt.show()
    autoencoder.save("./MODEL_25kp")
    encoder.save("./MODEL_25kp_encoder")
    decoder.save("./MODEL_25kp_decoder")
    
    test_image=x_train[0].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
    encoded_img1=encoder.predict(test_image)
    print("CODE SIZE ",encoded_img1.shape)
    #visu_skel(decoder.predict(encoded_img1),0)
    #visu_skel(x_train,0)
    
    y_actual=decoder.predict(encoded_img1)[0]
    y_pred=x_train[0]
    print("RMSE:",sqrt(mean_squared_error(y_actual, y_pred)))
    

    test_image=x_test[0].reshape(1,x_test[0].shape[0],x_train[0].shape[1])
    encoded_img1=encoder.predict(test_image)
    #visu_skel(decoder.predict(encoded_img1),0)
    #visu_skel(x_test,0)
    
    y_actual=decoder.predict(encoded_img1)[0]
    y_pred=x_test[0]
    print("RMSE:",sqrt(mean_squared_error(y_actual, y_pred)))
    #test_points(x_train,random=True,n=4)
    #latent_representation_tSNE(x_train,True)
    #visu_interpo(x_train,nb_iter,args.test,nb_frames,scaler,latent_size=25,show=False)
    index=0
    print("SAVING")
    for i in range(len(x_test)):
         encoded_img1=encoder.predict(x_test[i].reshape(1,x_test[0].shape[0],x_test[0].shape[1]))
         #print(scaler.inverse_transform(decoder.predict(encoded_img1).reshape(25,2)))
         #print(scaler.inverse_transform(decoder.predict(encoded_img1).reshape(25,2))[0])
         save_json_for_MocapNET(scaler.inverse_transform(decoder.predict(encoded_img1).reshape(25,2)),index,True)
         index+=1
         


