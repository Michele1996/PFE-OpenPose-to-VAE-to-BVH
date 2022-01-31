import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import os
import argparse

#skeleton BODY25
L = [[17,15],[15,0],[18,16],[16,0],[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,24],[11,22],[22,23],[8,12],[12,13],[13,14],[14,21],[14,19],[19,20]]



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

    
def visu_interpo(x_train,nb_iter,nb_frames,latent_size=2,show=True):
    num_files=0
    for o in range (nb_iter):
        test_image1=x_train[np.random.randint(0,len(x_train))].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
        test_image2=x_train[np.random.randint(0,len(x_train))].reshape(1,x_train[0].shape[0],x_train[0].shape[1])
        encoded_img1=encoder.predict(test_image1)
        encoded_img2=encoder.predict(test_image2)
        print("WOW")
        print(encoded_img1)
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
        np.random.seed(42)
        for i, image_idx in enumerate(interpolated_images):
            
            inter = interpolated_images[i].reshape(1,interpolated_images[i].shape[0])
            frame = decoder.predict(inter)[0]
            frame = interpolated_orig_images[i]
            #print(frame, len(frame))
            save_json_for_MocapNET(decoder.predict(inter)[0],num_files)
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

def save_json_for_MocapNET(frame,index):
    vector=""
    for i in range(len(frame)):
        if i%2==0 and i!=0:
           vector+=str(1)+","
        else:
           vector+=str(frame[i][0])+","
           vector+=str(frame[i][1])+","
    vector=vector[:len(vector)-1]
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
        json.dump(data_set, fp)
        fp.close()
    if(not os.path.isdir(path+"\\MocapNET-master")):
       print("Error, MocapNET not found. Please install it to continue")
       quit()
    if(not os.path.isdir(path+"\\MocapNET-master\\OUTPUT_to_BVH")):
        os.mkdir(path+"\\MocapNET-master\\OUTPUT_to_BVH")
    os.rename(os.path.join(path, filename), os.path.join(path+"\\MocapNET-master\\OUTPUT_to_BVH",filename))

def VAE(latent_size=25):
    ## Définition de l'architecture du modèle
    encoder_1_size = 256
    encoder_1_size_2 = 128
    latent_size = latent_size
    input_layer = tf.keras.layers.Input(shape = (25,2))
    flattened = tf.keras.layers.Flatten()(input_layer)
    encoder_1 = tf.keras.layers.Dense(encoder_1_size, activation = tf.keras.activations.linear)(flattened)
    encoder_1 = tf.keras.layers.BatchNormalization()(encoder_1)
    encoder_1 = tf.keras.layers.Dropout(0.3)(encoder_1)
    encoder_2 = tf.keras.layers.Dense(encoder_1_size_2, activation = 'relu')(encoder_1)
    encoder_2 = tf.keras.layers.BatchNormalization()(encoder_2)
    encoder_2 = tf.keras.layers.Dropout(0.3)(encoder_2)
    latent = tf.keras.layers.Dense(latent_size, activation = 'relu')(encoder_2)
    encoder = tf.keras.Model(inputs = input_layer, outputs = latent, name = 'encoder')
    #encoder.summary()

    input_layer_decoder = tf.keras.layers.Input(shape = encoder.output.shape[1:])
    decoder_1 = tf.keras.layers.Dense(encoder_1_size, activation = 'sigmoid')(input_layer_decoder)
    decoder_1 = tf.keras.layers.BatchNormalization()(decoder_1)
    decoder_1 = tf.keras.layers.Dropout(0.3)(decoder_1)
    decoder_2 = tf.keras.layers.Dense(encoder_1_size_2, activation = 'sigmoid')(decoder_1)
    decoder_2 = tf.keras.layers.BatchNormalization()(decoder_2)
    decoder_2 = tf.keras.layers.Dropout(0.3)(decoder_2)
    decoder_1 = tf.keras.layers.Dense(encoder.layers[1].output_shape[-1], activation = tf.keras.activations.linear)(decoder_2)
    constructed = tf.keras.layers.Reshape(x_train.shape[1:])(decoder_1)
    decoder = tf.keras.Model(inputs = input_layer_decoder, outputs = constructed, name= 'decoder')
    #decoder.summary()

    autoencoder = tf.keras.Model(inputs = encoder.input, outputs = decoder(encoder.output))
    autoencoder.summary()


    sgd = tf.keras.optimizers.Adam()
    autoencoder.compile(sgd, loss='mse', metrics=['accuracy'])
    return autoencoder,encoder,decoder

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("nb_iter", help="number of iter creation of nb_frames frames by encoder",type=int)
    parser.add_argument("nb_frames", help="number of frame create by encoder",type=int)
    parser.add_argument("data_file_location", help="path to data",type=str,nargs='?',default="keypoints_150_videos.npy")
    args = parser.parse_args()
    nb_frames=args.nb_frames
    nb_iter=args.nb_iter
    x_train=load_data(name =args.data_file_location)
    autoencoder,encoder,decoder = VAE(latent_size = 25)
    scaler = MinMaxScaler()
    # transform data
    for i in range(len(x_train)):
        x_train[i]= scaler.fit_transform(x_train[i])
    autoencoder.fit(x_train,x_train,batch_size=256,epochs=10)

    autoencoder.save("./MODEL_25kp")
    encoder.save("./MODEL_25kp_encoder")
    decoder.save("./MODEL_25kp_decoder")
    
    #simple_visu(x_train,0)
    #test_points(x_train,random=True,n=4)
    #latent_representation_tSNE(x_train,True)
    visu_interpo(x_train,nb_iter,nb_frames,latent_size=25,show=False)
    
    


