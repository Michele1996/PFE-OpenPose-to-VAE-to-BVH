import numpy
from BVHSmooth.code.helper import angles2quat, animate_rotations, display_animation, progressbar
from scipy import signal
from BVHSmooth.code.smooth_rotation import smooth
import argparse
import time

import sys

def slerp(one, two, t):
    """Spherical Linear intERPolation."""
    return (two * one.inverse())**t * one


parser = argparse.ArgumentParser()
parser.add_argument("input_file_gauche", help="path to input bvh file left arm",type=str)
parser.add_argument("input_file_droit", help="path to input bvh file right arm",type=str)
args = parser.parse_args()
input_filename=args.input_file_gauche
input_filename2=args.input_file_droit
#read input file
fin = open(input_filename, "rt")
fin1 = open(input_filename2, "rt")
#read file contents to string
data = fin.read()
data1 = fin1.read()
#replace all occurrences of the required string
data = data.replace('head', 'Head')
data = data.replace('neck', 'Neck')
data = data.replace('Neck1', 'neck1')
data = data.replace('chest', 'Chest2')
data = data.replace('abdomen', 'Chest')
data = data.replace('rCollar', 'RightCollar')
data = data.replace('lCollar', 'LeftCollar')
data = data.replace('rShldr', 'RightShoulder')
data = data.replace('lShldr', 'LeftShoulder')
data = data.replace('rForeArm', 'RightElbow')
data = data.replace('lForeArm', 'LeftElbow')
data = data.replace('rHand', 'RightWrist')
data = data.replace('lHand', 'LeftWrist')

data1 = data1.replace('head', 'Head')
data1 = data1.replace('neck', 'Neck')
data1 = data1.replace('Neck1', 'neck1')
data1 = data1.replace('chest', 'Chest2')
data1 = data1.replace('abdomen', 'Chest')
data1 = data1.replace('rCollar', 'RightCollar')
data1 = data1.replace('lCollar', 'LeftCollar')
data1 = data1.replace('rShldr', 'RightShoulder')
data1 = data1.replace('lShldr', 'LeftShoulder')
data1 = data1.replace('rForeArm', 'RightElbow')
data1 = data1.replace('lForeArm', 'LeftElbow')
data1 = data1.replace('rHand', 'RightWrist')
data1 = data1.replace('lHand', 'LeftWrist')
#close the input file
fin.close()
fin1.close()
#open the input file in write mode
fin = open(input_filename, "wt")
fin1 = open(input_filename2, "wt")
#overrite the input file with the resulting data
fin.write(data)
fin1.write(data1)
#close the file
fin.close()
fin1.close()

#old_bvh=open("test3.bvh","r")

old_bvh=open(input_filename,"r")
old_bvh1=open(input_filename2,"r")
#new_bvh=open("new_bvh_test1.bvh","w")
output_filename="new_bvh_"+input_filename
new_bvh=open(output_filename,"w")
l=old_bvh.readlines()
l1=old_bvh1.readlines()
not_to_change=["Knee","Ankle","Foot"]
lines=[]
index=0
motion_index=-1
motion_lines=[]
###########
for line in l:
    if("JOINT" in line or "ROOT" in line):
        lines.append((line.replace(" ","").replace("JOINT","").replace("ROOT","").replace("\t","").replace("\n",""),index))
        index+=1
        #print(lines[index-1])
    if(motion_index>=0):
        motion_lines.append(line.split())
        #print(motion_lines[motion_index])
        motion_index+=1
    if("Frame Time" in line):
        motion_index+=1
###########
lines1=[]
index1=0
motion_index1=-1
motion_lines1=[]
for line in l1:
    if("JOINT" in line or "ROOT" in line):
        lines1.append((line.replace(" ","").replace("JOINT","").replace("ROOT","").replace("\t","").replace("\n",""),index1))
        index1+=1
    if(motion_index1>=0):
        motion_lines1.append(line.split())
        #print(motion_lines[motion_index])
        motion_index1+=1
    if("Frame Time" in line):
        motion_index1+=1


index_collar=0
index_shoulder=0
index_elbow=0
index_wrist=0

for p in range(len(motion_lines1)):
    for j in lines1:
        if("Left" in j[0]):
            if("Collar" in j[0]):
                index_collar=j[1]
            if("Shoulder" in j[0]):
                index_shoulder=j[1]
            if("Elbow" in j[0]):
                index_elbow=j[1]
            if("Wrist" in j[0]):
                index_wrist=j[1]
print(index_collar, index_shoulder,index_elbow,index_wrist)
for p in range(len(motion_lines)): 
    #print("Processing FRAME:"+str(p+1)+"/"+str(len(motion_lines)))
    for i in lines:
        if("Head" in i[0]) :
           #print(motion_lines[p][i[1]*3+3], motion_lines[p][i[1]*3+3+1],motion_lines[p][i[1]*3+3+2])
           a="-"+motion_lines[p][i[1]*3+3]
           motion_lines[p][i[1]*3+3]=motion_lines[p][i[1]*3+3+1]
           motion_lines[p][i[1]*3+3+1]=a
        if("Right" in i[0]):
           if("Collar" in i[0]):
               #print(i)
               motion_lines[p][i[1]*3+3]=str(round(-float(motion_lines1[p][index_collar*3+3]),2))
               motion_lines[p][i[1]*3+3+1]=str(round(float(motion_lines1[p][index_collar*3+3+1]),2)) 
               motion_lines[p][i[1]*3+3+2]=str(round(-float(motion_lines1[p][index_collar*3+3+2]),2))
           if("Shoulder" in i[0]):
               motion_lines[p][i[1]*3+3]=str(round(-float(motion_lines1[p][index_shoulder*3+3]),2))
               motion_lines[p][i[1]*3+3+1]=str(round(float(motion_lines1[p][index_shoulder*3+3+1]),2))
               motion_lines[p][i[1]*3+3+2]=str(round(-float(motion_lines1[p][index_shoulder*3+3+2]),2))
           elif("Elbow" in i[0]):
               motion_lines[p][i[1]*3+3]=str(round(-float(motion_lines1[p][index_elbow*3+3]),2))
               if(float(motion_lines[p][i[1]*3+3])<-20):
                  motion_lines[p][i[1]*3+3]="-20"
               motion_lines[p][i[1]*3+3+1]=str(round(float(motion_lines1[p][index_elbow*3+3+1]),2))
               if(float(motion_lines[p][i[1]*3+3+1])<-30):
                  motion_lines[p][i[1]*3+3+1]="-30"
               motion_lines[p][i[1]*3+3+2]=str(round(-float(motion_lines1[p][index_elbow*3+3+2]),2))
               if(float(motion_lines[p][i[1]*3+3+2])<-20):
                  motion_lines[p][i[1]*3+3+2]="-20"
           elif("Wrist"in i[0]):
               motion_lines[p][i[1]*3+3]=str(round(-float(motion_lines1[p][index_wrist*3+3]),2))
               motion_lines[p][i[1]*3+3+1]=str(round(float(motion_lines1[p][index_wrist*3+3+1]),2))
               motion_lines[p][i[1]*3+3+2]=str(round(-float(motion_lines1[p][index_wrist*3+3+2]),2))
        elif("Left" in i[0]):
           if("Collar" in i[0]):
               #print(i)
               motion_lines[p][i[1]*3+3]=str(round(float(motion_lines[p][i[1]*3+3]),2))
               motion_lines[p][i[1]*3+3+1]=str(round(float(motion_lines[p][i[1]*3+3+1]),2))
               motion_lines[p][i[1]*3+3+2]=str(round(float(motion_lines[p][i[1]*3+3+2]),2))
           if("Shoulder" in i[0]):
               motion_lines[p][i[1]*3+3]=str(round(float(motion_lines[p][i[1]*3+3]),2))
               motion_lines[p][i[1]*3+3+1]=str(-round(float(motion_lines[p][i[1]*3+3+1]),2))
               motion_lines[p][i[1]*3+3+2]=str(round(-float(motion_lines[p][i[1]*3+3+2]),2))
           elif("Elbow" in i[0]):
               motion_lines[p][i[1]*3+3]=str(round(-float(motion_lines[p][i[1]*3+3]),2))
               motion_lines[p][i[1]*3+3+1]=str(round(float(motion_lines[p][i[1]*3+3+1]),2))
               motion_lines[p][i[1]*3+3+2]=str(round(-float(motion_lines[p][i[1]*3+3+2]),2))
           elif("Wrist"in i[0]):
               motion_lines[p][i[1]*3+3]=str(round(float(motion_lines[p][i[1]*3+3]),2))
               motion_lines[p][i[1]*3+3+1]=str(round(45+float(motion_lines[p][i[1]*3+3+1]),2))
               motion_lines[p][i[1]*3+3+2]=str(round(+float(motion_lines[p][i[1]*3+3+2]),2))
            
            
        elif("Left" not in i[0] and "Right" not in i[0] and "Head" not in i[0] or "Foot" in i[0] or "Ankle" in i[0] or "Knee" in i[0] or "Hip" in i[0] ):
           #print(i)
           motion_lines[p][i[1]*3+3]=str(0)
           motion_lines[p][i[1]*3+3+1]=str(0)
           motion_lines[p][i[1]*3+3+2]=str(0)
           if("Hips" in i[0]):
               motion_lines[p][i[1]]=str(0)
               motion_lines[p][i[1]+1]=str(0)
               motion_lines[p][i[1]+2]=str(0)
            
               
       
for line in l:
    if("Frames" in line):
        new_bvh.write(line.replace("300","1200"))
    elif("Frame Time" in line):
        new_bvh.write(line.replace("0.04","0.016667"))
        break
    else:
        new_bvh.write(line)
frames=[]

for k in progressbar(range(len(motion_lines)-1), "SLERP (2 STEPS): ", 40):
    #print("PROCESSING FRAME "+str(k+1)+"/"+str(len(motion_lines)))
    start=motion_lines[k]
    end=motion_lines[k+1]
    #use 4 best
    for i in range(4):
        stringa=str(motion_lines[k][0])+" "+str(motion_lines[k][1])+" "+str(motion_lines[k][2])+" "
        for j in range(0,len(motion_lines[0])-3,3):
            angles_start=[round(float(motion_lines[k][j+3+1]),2),round(float(motion_lines[k][j+3+2]),2),round(float(motion_lines[k][j+3]),2)]
            angles_end=[round(float(motion_lines[k+1][j+3+1]),2),round(float(motion_lines[k+1][j+3+2]),2),round(float(motion_lines[k+1][j+3]),2)]
            q1 = angles2quat(angles_start[0],angles_start[1],angles_start[2])
            q2 = angles2quat(angles_end[0],angles_end[1],angles_end[2])
            ani_times = numpy.linspace(0, 1, 2)
            p=slerp(q1,q2,ani_times)
            #in order to be have the same orientation as GRETA we need to put z=-z and y=-y
            stringa=stringa+str(round(-angles_start[2]))+" "+str(round(angles_start[0]))+" "+str(-round(angles_start[1]))+" "
            #stringa=stringa+str(round(angles_start[2]))+" "+str(round(angles_start[0]))+" "+str(round(angles_start[1]))+" "
        stringa+="\n"
        frames.append(stringa.split())
modifs=[]
d=7
for i in progressbar(range(len(frames[0])), "APPLYING MEDIAN FILTER ON EACH VALUE OF ALL THE FRAMES: ", 40):
    x=frames[0:len(frames)]
    modif=[]
    for j in range(len(x)):
        modif.append(float(x[j][i]))
    modif2=signal.medfilt(modif, d)
    modifs.append(modif2)
for i in progressbar(range(len(modifs[0])), "CREATE FILE WITH OLD AND NEW GENERATED FRAMES: ", 40):
    stringa=""
    for j in range(len(modifs)):
        stringa+=str(modifs[j][i])+" "
    stringa=stringa+"\n"
    new_bvh.write(stringa) 
input=output_filename
smooth(input,input.replace(".bvh","_smooth.bvh"))