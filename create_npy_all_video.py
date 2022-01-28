import pandas as pd
import os
import numpy as np

def create_npy(path="./output",scene_len = None):
    if scene_len == "max":
        maximum = 0
        for video in os.listdir(path):
            length = np.max(len(os.listdir(path+"\\"+video))-1)
            if maximum <  length:
                maximum = length
        if maximum%2 == 1 :
            maximum +=1
    elif type(scene_len) == int :
        maximum = scene_len
        if maximum%2 == 1 :
            maximum +=1
    else :
        maximum = -1

    kp_total = []
    for video in os.listdir(path):
        kp_video=[]
        for frame in os.listdir(path+"\\"+video):
            if(len(os.listdir(path+"\\"+video)) > 0) :
                print(path+"\\"+video+"\\"+frame)
                df = pd.read_json(path+"\\"+video+"\\"+frame)
                people = df["people"]
                print(df["people"])
                kp = people[0]["pose_keypoints_2d"]
                kp = np.delete(kp,[i for i in range(2,len(kp),3)])
                #print(kp)
                kp_total.append(kp)
    #print(len(kp_total))
            if maximum >= 0 :
                temp = np.zeros((maximum,50))
                temp[:len(kp_video)] = kp_video
                kp_video = temp
        
            if np.size(kp_video) > 0:
                kp_total.append(np.array(kp_video))
    np.save("kp_all_videos",np.array(kp_total,dtype=object))


create_npy()


