import cv2
import glob
import os
import argparse
import numpy as np
from scipy.linalg import norm
from scipy import sum, average

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng
     
def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    dist_euclidean = np.sqrt(np.sum((diff)**2)) / img1.size
    return (m_norm, z_norm, dist_euclidean)

list_file=glob.glob("videos/scenes/*.mp4")
nb_similar_frame=0
nb_frame=0
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", required=False, default=0.0001)
args = parser.parse_args()
print(args.threshold)
args=float(args.threshold)
for path in list_file:
    cap = cv2.VideoCapture(path)
    print(path)
    ret, current_frame = cap.read()
    previous_frame = current_frame
    nb_similar_frame=0
    nb_frame=0
    while(cap.isOpened()):
        #print(current_frame)
        if(not ret):
            break
        else:
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY) 
            n_m, n_0,n_2 = compare_images(current_frame_gray, previous_frame_gray)
            #print ("Manhattan norm:", n_m, "/ per pixel:", n_m/current_frame_gray.size)
            #print ("Zero norm:", n_0, "/ per pixel:", n_0*1.0/current_frame_gray.size)
            #print ("L2 norm:", n_2, "/ per pixel:", n_2*1.0/current_frame_gray.size)
            if(n_2<args):
               nb_similar_frame+=1
            previous_frame = current_frame.copy()
            ret, current_frame = cap.read()
            nb_frame+=1
    cap.release()
    print(nb_similar_frame)
    if(nb_frame < 10):
        print("Too short scene:", path)
        os.remove(path)
    if(nb_similar_frame>(nb_frame*20)/100):
        print("is a picture:", path)
        os.remove(path)


 