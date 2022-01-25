import pandas as pd
import os
import numpy as np
import argparse

def create_npy(path=".output/output_final/",scene_len=None):
    if scene_len == "max":
        maximum = 0
        for scene in os.listdir(path):
            length = np.max(len(os.listdir(path+scene))-1)
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
    for scene in os.listdir(path):
        kp_scene=[]
        for frame in os.listdir(path+scene):
            if frame[-4:]!='.avi' and len(os.listdir(path+scene)) > 0 :
                #print(path+scene+"/"+frame)
                df = pd.read_json(path+scene+"/"+frame)
                people = df["people"]
                kp = people[0]["pose_keypoints_2d"]
                kp = np.delete(kp,[i for i in range(2,len(kp),3)])
                #print(kp)
                kp_scene.append(kp)

        if maximum >= 0 :
            temp = np.zeros((maximum,50))
            temp[:len(kp_scene)] = kp_scene
            kp_scene = temp
        if np.size(kp_scene) > 0:
            kp_total.append(np.array(kp_scene))
    np.save("kp_3",np.array(kp_total,dtype=object))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_output", help="path to openpose json output",type=str)
    parser.add_argument("len_scene", help="max length for each scene",type=int)
    args = parser.parse_args()
    nb_iter=args.nb_iter
    create_npy(args.path_to_output, args.len_scene)


