import json
import os
import glob
import numpy as np

def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
#path='video-Scene-001_000000000005_keypoints.json'
list_file_to_discard=[]
list_file=glob.glob("output/*.json")
keep=True
i=0
for filename in list_file:
        i+=1
        with open(filename) as f:
          data = json.load(f)
          #print(data, len(data["people"]))
          if(len(data["people"])!=1):
                print("More than one person in this scene or no one in the scen, discard")
                if(len(data["people"])>0):
                    #print(data["people"][0]["pose_keypoints_2d"][0],data["people"][0]["pose_keypoints_2d"][1], data["people"][0]["pose_keypoints_2d"][14*3],data["people"][0]["pose_keypoints_2d"][14*3+1])
                    dist=distance(data["people"][0]["pose_keypoints_2d"][0],data["people"][0]["pose_keypoints_2d"][1],data["people"][0]["pose_keypoints_2d"][14*3],data["people"][0]["pose_keypoints_2d"][14*3+1])
                   # print("Distanza:",dist)
                   # print(data["people"][0]["pose_keypoints_2d"][4*3]-data["people"][0]["pose_keypoints_2d"][7*3]>0)
                    dist=distance(data["people"][0]["pose_keypoints_2d"][4*3],data["people"][0]["pose_keypoints_2d"][4*3+1],data["people"][0]["pose_keypoints_2d"][7*3],data["people"][0]["pose_keypoints_2d"][7*3+1])
                    #print(dist)
                if(filename not in list_file_to_discard):
                    list_file_to_discard.append(filename)
                    keep=False
          else:
                dist=distance(data["people"][0]["pose_keypoints_2d"][0],data["people"][0]["pose_keypoints_2d"][1],data["people"][0]["pose_keypoints_2d"][14*3],data["people"][0]["pose_keypoints_2d"][14*3+1])
                #print(dist)
                if(dist<150):
                    
                    print("Small Skeleton, discard scene",dist)
                    if(filename not in list_file_to_discard):
                       list_file_to_discard.append(filename)
                       keep=False
                dist=distance(data["people"][0]["pose_keypoints_2d"][4*3],data["people"][0]["pose_keypoints_2d"][4*3+1],data["people"][0]["pose_keypoints_2d"][7*3],data["people"][0]["pose_keypoints_2d"][7*3+1])
                #print(dist)
                if(dist<0):
                    print("Camera on the back, discard scene")
                    if(filename not in list_file_to_discard):
                       list_file_to_discard.append(filename)
                       keep=False
                for i in range(0,len(data["people"][0]["pose_keypoints_2d"]),3):
                    if(data["people"][0]["pose_keypoints_2d"][i]==0):
                        print("Missing keypoint")
                        keep=False
                for i in range(1,len(data["people"][0]["pose_keypoints_2d"]),3):
                    if(data["people"][0]["pose_keypoints_2d"][i]==0):
                        print("Missing keypoint")
                        keep=False
                    
          if(keep):
             print("keep "+filename,i)
          else:
              print("Delete ",filename)
              f.close()
              os.remove(filename)
          keep=True
               
print(len(list_file_to_discard))