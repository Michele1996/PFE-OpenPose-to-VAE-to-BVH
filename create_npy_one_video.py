import pandas as pd
import os
import numpy as np

def create_npy(path="./output/0",scene_len = None):
    if scene_len == "max":
        maximum = 0
        for scene in os.listdir(path):
            length = np.max(len(os.listdir(path))-1)
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
    for frame in os.listdir(path):
        if(len(os.listdir(path)) > 0) :
            print(frame)
            df = pd.read_json(path+"\\"+frame)
            people = df["people"]
            print(df["people"])
            kp = people[0]["pose_keypoints_2d"]
            kp = np.delete(kp,[i for i in range(2,len(kp),3)])
            #print(kp)
            kp_total.append(kp)
    #print(len(kp_total))
    np.save("kp"+path.replace(".","").replace("/","-"),np.array(kp_total,dtype=object))

create_npy()


