import pickle
from pickle import dump, load
import numpy as np
import os
import json
import sys
import matplotlib.pyplot as plt
L = [[10,2],[2,1]]

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def get_bone_list():
    bone_list = [[0, 1], [1, 2], [2, 10], [6, 7], [6, 8], [3, 4], [4, 5], [5, 9], [-1, 0], [-1, 3], [-1, 6]]
    #bone_list = [[1, 2], [2, 3], [3, 11], [7, 8], [7, 9], [4, 5], [5, 6], [-1, 1], [-1, 4], [-1, 7], [6, 10]]
    #bone_list = [[1, 2], [2, 3], [3, 11], [7, 8], [7, 9], [4, 5], [5, 6], [6, 10]]
    return bone_list
    
def visu_skel(x_train):
    frame = x_train
    plt.figure()
    plt.xlim(left = -1000,right=1000)
    plt.ylim(top=1000,bottom=-1000)
    plt.scatter(frame[0,:],-frame[1,:],s=10)
    bone_list = get_bone_list()
    x = frame[0, :]
    y = -frame[1, :]
    for bone in bone_list:
                joint1_id = bone[0]
                joint2_id = bone[1]
                plt.plot([x[joint1_id], x[joint2_id]], [y[joint1_id], y[joint2_id]])
                
    plt.show()
def save_json_for_MocapNET(frame,index,left_hand,right_hand,index_ftest=False):
    vector=[]
    #print("PROCESS FRAME: ",index_f, index)
    #print(frame)
    #print(len(frame))
    for i in range(0,len(frame)-1,2):
           vector.append(round(float(frame[i]),2))
           vector.append(round(float(frame[i+1]),2))
           vector.append(float(0.80))
    vector_left=[]
    for i in range(0,len(left_hand)-1,2):
           vector_left.append(round(float(left_hand[i]),2))
           vector_left.append(round(float(left_hand[i+1]),2))
           vector_left.append(float(0.80))
    vector_right=[]
    for i in range(0,len(right_hand)-1,2):
           vector_right.append(round(float(right_hand[i]),2))
           vector_right.append(round(float(right_hand[i+1]),2))
           vector_right.append(float(0.80))
    #print("VETTORE ",len(vector))
    data_set = {"version": 1.3, "people": [{"person_id":[-1],"pose_keypoints_2d":vector,"face_keypoints_2d":[],"hand_left_keypoints_2d":vector_left,"hand_right_keypoints_2d":vector_right,"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}
    index="0000"+str(index)
    decalage=len(str(index))-5
    index=index[decalage:len(str(index))]
    path = os.getcwd()
    filename="colorFrame_"+str(index_f)+"_"+str(index)+"_keypoints.json"
    if os.path.exists(path+"\\Test2\\"+filename):
          os.remove(path+"\\Test2\\"+filename)
    if(not os.path.isdir(path+"\\Test2")):
            os.mkdir(path+"\\Test2")
    with open(filename, 'w') as fp:
        json.dump(data_set, fp,separators=(',', ':'))
        fp.close()
    os.rename(os.path.join(path, filename), os.path.join(path+"\\Test2\\",filename))
        
        
def save_json_for_MocapNET_10k(frame,ind,droite_a_gauche=False,test=False):
    vector=[]
    #print("PROCESS FRAME: ",index_f, index)
    #print(frame)
    #print(len(frame))
    for i in range(0,len(frame)-1,2):
           vector.append(round(float(frame[i]),2))
           vector.append(round(float(frame[i+1]),2))
           vector.append(float(0.80))
    #print("VETTORE ",len(vector))
    data_set = {"version": 1.3, "people": [{"person_id":[-1],"pose_keypoints_2d":vector,"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}
    ind="0000"+str(ind)
    decalage=len(str(ind))-5
    ind=ind[decalage:len(str(ind))]
    path = os.getcwd()
    filename="colorFrame_"+str(0)+"_"+str(ind)+"_keypoints.json"
    folder_name="\\Test_Gauche\\"
    if(droite_a_gauche):
        folder_name="\\Test_Droite\\"
    if os.path.exists(path+folder_name+filename):
          os.remove(path+folder_name+filename)
    if(not os.path.isdir(path+folder_name)):
            os.mkdir(path+folder_name)
    with open(filename, 'w') as fp:
        json.dump(data_set, fp,separators=(',', ':'))
        fp.close()
    os.rename(os.path.join(path, filename), os.path.join(path+folder_name,filename))
# Loading Database
#pose_train = load(open('colbert_batch_pose.p', 'rb'))
pose_train = load(open('cmu0000031081_new.p', 'rb'))
print(pose_train.shape)
# 'pose/data" has 104 dimensions which is the same as 52 joints with XY coordinates. Let's reshape it to a more obvious format.

#print(pose_train.shape)
#print(pose_train[0][0].shape)
#print(pose_train[0][0])

"""
index_f=0
#print(len(pose_train), len(pose_train[0]), len(pose_train[0][0]), len(pose_train[0][0][0]))
for i in progressbar(range(len(pose_train)), "CREATE JSON OPENPOSE_FORMAT: ", 40):
    index=0
    for j in range(len(pose_train[0])):
        pose_x=[]
        pose_y=[]
        hand_left_x=[]
        hand_left_y=[]
        hand_right_x=[]
        hand_right_y=[]
        new_frame_body=[]
        new_frame_left_hand=[]
        new_frame_right_hand=[]
        for k in range(10):
            pose_x.append(float(pose_train[i][j][0][k]))
            pose_y.append(float(pose_train[i][j][1][k]))
        for k in range(10,30):
            hand_left_x.append(float(pose_train[i][j][0][k]))
            hand_left_y.append(float(pose_train[i][j][1][k]))
        for k in range(30,len(pose_train[0][0][0])):
            hand_right_x.append(float(pose_train[i][j][0][k]))
            hand_right_y.append(float(pose_train[i][j][1][k]))
        for l in range(len(pose_x)):
            new_frame_body.append(pose_x[l])
            new_frame_body.append(pose_y[l])
        for z in range(len(hand_left_x)):
            new_frame_left_hand.append(hand_left_x[z])
            new_frame_left_hand.append(hand_left_y[z])
        for p in range(len(hand_right_x)):
            new_frame_right_hand.append(hand_right_x[p])
            new_frame_right_hand.append(hand_right_y[p])
        #print(len(new_frame_body),index,len(new_frame_left_hand),index_f)
        save_json_for_MocapNET(new_frame_body,index,new_frame_left_hand,new_frame_right_hand,index_f)
        index=index+1
    index_f=index_f+1

"""
"""
#Format Mireille
index ={"1":0,"5":1,"6":2,"7":3,"2":4,"3":5,"4":6,"0":7,"16":8,"15":9}
index_f=0
print(len(pose_train[0][0]))
ind=0
for i in progressbar(range(len(pose_train[0])), "CREATE JSON OPENPOSE_FORMAT: ", 40):
    for j in range(64):
        pose_25=[]
        pose_x=[]
        pose_y=[]
        for k in range(12):
            #print(pose_train[i][j][0][k])
            pose_x.append(float(pose_train[0][i][j][0][k]))
            pose_y.append(float(pose_train[0][i][j][1][k]))
        for p in range(25):
            if(str(p) in index.keys()):
                    pose_25.append(pose_x[index[str(p)]])
                    pose_25.append(pose_y[index[str(p)]])
            else:
                pose_25.append(0)
                pose_25.append(0)
        save_json_for_MocapNET_10k(pose_25,ind)
        ind=ind+1
"""
### Format Mireille 2
index ={"1":-1,"5":0,"6":1,"7":2,"2":3,"3":4,"4":5,"0":6,"16":8,"15":7}
#index ={"1":0,"5":1,"6":2,"7":3,"2":4,"3":5,"4":6,"0":7,"16":8,"15":9}
index_droit=["2","3","4"]
index_f=0
print(len(pose_train[0]))
ind=0
for i in progressbar(range(len(pose_train)), "CREATE JSON OPENPOSE_FORMAT: ", 40):
        pose_25=[]
        pose_x=[]
        pose_y=[]
        pose_12=[]
        for k in range(12):
            #print(pose_train[i][j][0][k])
            pose_x.append(float(pose_train[i][0][k]))
            pose_y.append(float(pose_train[i][1][k]))
        for p in range(25):
            if(str(p) in index.keys()):
                   #if(p>=5 and p<=7):
                      #print(str(p)+" Gauche "+str(pose_x[index[str(p)]])+"  "+str(pose_y[index[str(p)]]))
                   #elif(p>=2 and p<=4):
                      #print(str(p)+" Droite "+str(pose_x[index[str(p)]])+"  "+str(pose_y[index[str(p)]]))
                   pose_25.append(pose_x[index[str(p)]])
                   pose_25.append(pose_y[index[str(p)]])
            else:
                pose_25.append(0)
                pose_25.append(0)
        #visu_skel(pose_train[i])
        save_json_for_MocapNET_10k(pose_25,ind)
        ind=ind+1
print("Finished Gauche Gauche")
ind=0
index ={"1":-1,"5":3,"6":4,"7":5,"2":3,"3":4,"4":5,"0":6,"16":8,"15":7}
for i in progressbar(range(len(pose_train)), "CREATE JSON OPENPOSE_FORMAT: ", 40):
        pose_25=[]
        pose_x=[]
        pose_y=[]
        pose_12=[]
        for k in range(12):
            #print(pose_train[i][j][0][k])
            pose_x.append(float(pose_train[i][0][k]))
            pose_y.append(float(pose_train[i][1][k]))
        for p in range(25):
            if(str(p) in index.keys()):
                   if(p>=5 and p<=7):
                       pose_25.append(-pose_x[index[str(p)]])
                       pose_25.append(pose_y[index[str(p)]])
                   else:
                       pose_25.append(pose_x[index[str(p)]])
                       pose_25.append(pose_y[index[str(p)]])
            else:
                pose_25.append(0)
                pose_25.append(0)
        #visu_skel(pose_train[i])
        save_json_for_MocapNET_10k(pose_25,ind, True)
        ind=ind+1
print("Finished Droite Gauche")