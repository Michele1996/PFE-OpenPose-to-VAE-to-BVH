import json
import os
import glob
import numpy as np

def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
list_file_to_discard=[]
path="output/"
list_file=glob.glob("output/*.json")
keep=True
for filename in list_file:
        with open(filename) as f:
          data = json.load(f)
          if(len(data["people"])!=1):
                if(filename not in list_file_to_discard):
                    list_file_to_discard.append(filename)
                    keep=False
                    f.close()
                    if(os.path.isfile(path+filename)):
                        print("More than one person in this scene or no one in the scen, discard")
                        os.remove(filename)
          else:
                dist=distance(data["people"][0]["pose_keypoints_2d"][0],data["people"][0]["pose_keypoints_2d"][1],data["people"][0]["pose_keypoints_2d"][14*3],data["people"][0]["pose_keypoints_2d"][14*3+1])
                if(dist<150):
                    if(filename not in list_file_to_discard):
                       list_file_to_discard.append(filename)
                       keep=False
                       f.close()
                       if(os.path.isfile(path+filename)):
                          print("Small Skeleton, discard scene",dist)
                          os.remove(filename)
                dist=distance(data["people"][0]["pose_keypoints_2d"][4*3],data["people"][0]["pose_keypoints_2d"][4*3+1],data["people"][0]["pose_keypoints_2d"][7*3],data["people"][0]["pose_keypoints_2d"][7*3+1])
                #print(dist)
                if(dist<0):
                    if(filename not in list_file_to_discard):
                       list_file_to_discard.append(filename)
                       keep=False
                       f.close()
                       if(os.path.isfile(path+filename)):
                          print("Camera on the back, discard scene")
                          os.remove(filename)
                for i in range(0,len(data["people"][0]["pose_keypoints_2d"]),3):
                    if(data["people"][0]["pose_keypoints_2d"][i]==0):
                        keep=False
                        f.close()
                        if(os.path.isfile(path+filename)):
                           print("Missing keypoint")
                           os.remove(filename)
                for i in range(1,len(data["people"][0]["pose_keypoints_2d"]),3):
                    if(data["people"][0]["pose_keypoints_2d"][i]==0):
                        keep=False
                        f.close()
                        if(os.path.isfile(path+filename)):
                            print("Missing keypoint")
                            os.remove(filename)
                    
          if(keep):
             print("keep "+filename)
          else:
             print("Delete ",filename)
          keep=True
               
print(len(list_file_to_discard))