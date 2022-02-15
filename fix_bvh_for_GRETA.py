import numpy
from helper import angles2quat, animate_rotations, display_animation
from scipy import signal
from BVHSmooth.code.smooth_rotation import smooth
import argparse
def slerp(one, two, t):
    """Spherical Linear intERPolation."""
    return (two * one.inverse())**t * one


parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="path to input bvh file",type=str)
args = parser.parse_args()
input_filename=args.input_file
#read input file
fin = open(input_filename, "rt")
#read file contents to string
data = fin.read()
#replace all occurrences of the required string
data = data.replace('head', 'Head')
data = data.replace('neck', 'Neck')
data = data.replace('chest', 'Chest2')
data = data.replace('abdomen', 'Chest')
data = data.replace('rCollar', 'RightCollar')
data = data.replace('lCollar', 'LeftCollar')
data = data.replace('rForeArm', 'RightElbow')
data = data.replace('lForeArm', 'LeftElbow')
data = data.replace('rHand', 'RightWrist')
data = data.replace('lHand', 'LeftWrist')
#close the input file
fin.close()
#open the input file in write mode
fin = open(input_filename, "wt")
#overrite the input file with the resulting data
fin.write(data)
#close the file
fin.close()

#old_bvh=open("test3.bvh","r")

old_bvh=open(input_filename,"r")
#new_bvh=open("new_bvh_test1.bvh","w")
output_filename="new_bvh_"+input_filename
new_bvh=open(output_filename,"w")
l=old_bvh.readlines()
not_to_change=["Knee","Ankle","Foot"]
lines=[]
index=0
motion_index=-1
motion_lines=[]
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
#reverse right and left collar, shoulder,elbow and wrist values

for p in range(len(motion_lines)): 
    #print("Processing FRAME:"+str(p+1)+"/"+str(len(motion_lines)))
    for i in lines:
        if("Left" in i[0] and i[0].replace("Left","") not in not_to_change):
            #print(i)
            #print(motion_lines[p][i[1]*3+3])
            #print(motion_lines[p][i[1]*3+3+1])
            #print(motion_lines[p][i[1]*3+3+2])
            val_1=motion_lines[p][i[1]*3+3]
            val_2=motion_lines[p][i[1]*3+3+1]
            val_3=motion_lines[p][i[1]*3+3+2]
            for j in lines:
                if(j[0]==i[0].replace("Left","Right")):
                   #print(j)
                   #print(motion_lines[p][j[1]*3+3])
                   #print(motion_lines[p][j[1]*3+3+1])
                   #print(motion_lines[p][j[1]*3+3+2])
                   if("Collar" in j[0]):
                       #print(i)
                       motion_lines[p][i[1]*3+3]=str(round(float(motion_lines[p][j[1]*3+3]),2))
                       motion_lines[p][i[1]*3+3+1]=str(round(float(motion_lines[p][j[1]*3+3+1]),2))
                       motion_lines[p][i[1]*3+3+2]=str(round(float(motion_lines[p][j[1]*3+3+2]),2))
                       motion_lines[p][j[1]*3+3]=str(round(float(val_1),2))
                       motion_lines[p][j[1]*3+3+1]=str(round(float(val_2),2))
                       motion_lines[p][j[1]*3+3+2]=str(round(float(val_3),2))
                   if("Shoulder" in j[0]):
                       motion_lines[p][i[1]*3+3]=str(round(float(motion_lines[p][j[1]*3+3]),2))
                       motion_lines[p][i[1]*3+3+1]=str(round(float(motion_lines[p][j[1]*3+3+1])-60,2))
                       motion_lines[p][i[1]*3+3+2]=str(round(float(motion_lines[p][j[1]*3+3+2])-60,2))
                       motion_lines[p][j[1]*3+3]=str(round(float(val_1),2))
                       motion_lines[p][j[1]*3+3+1]=str(round(float(val_2),2))
                       motion_lines[p][j[1]*3+3+2]=str(round(float(val_3),2))
                   elif("Elbow" in j[0]):
                       motion_lines[p][i[1]*3+3]=str(round(float(motion_lines[p][j[1]*3+3]),2))
                       motion_lines[p][i[1]*3+3+1]=str(round(float(motion_lines[p][j[1]*3+3+1])+150,2))
                       motion_lines[p][i[1]*3+3+2]=str(round(float(motion_lines[p][j[1]*3+3+2])+0,2))
                       motion_lines[p][j[1]*3+3]=str(round(float(val_1),2))
                       motion_lines[p][j[1]*3+3+1]=str(round(float(val_2),2))
                       motion_lines[p][j[1]*3+3+2]=str(round(float(val_3),2))
                   elif("Wrist"in j[0]):
                       motion_lines[p][i[1]*3+3]=str(round(float(motion_lines[p][j[1]*3+3]),2))
                       motion_lines[p][i[1]*3+3+1]=str(round(float(motion_lines[p][j[1]*3+3+1])+90,2))
                       motion_lines[p][i[1]*3+3+2]=str(round(float(motion_lines[p][j[1]*3+3+2]),2))
                       motion_lines[p][j[1]*3+3]=str(round(float(val_1),2))
                       motion_lines[p][j[1]*3+3+1]=str(round(float(val_2),2))
                       motion_lines[p][j[1]*3+3+2]=str(round(float(val_3),2))
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
        x=line.split()
        new_bvh.write(line.replace(x[1],str(int(x[1])*5)))
    elif("Frame Time" in line):
        new_bvh.write(line.replace("0.04","0.016667"))
        break
    else:
        new_bvh.write(line)
frames=[]
for k in range(len(motion_lines)-1):
    print("PROCESSING FRAME "+str(k+1)+"/"+str(len(motion_lines)))
    #print(len(motion_lines[0]))
    #print(motion_lines[0])
    start=motion_lines[k]
    end=motion_lines[k+1]
    #use 4 best
    for i in range(3):
        stringa=str(motion_lines[k][0])+" "+str(motion_lines[k][1])+" "+str(motion_lines[k][2])+" "
        for j in range(0,len(motion_lines[0])-3,3):
            angles_start=[round(float(motion_lines[k][j+3+1]),2),round(float(motion_lines[k][j+3+2]),2),round(float(motion_lines[k][j+3]),2)]
            angles_end=[round(float(motion_lines[k+1][j+3+1]),2),round(float(motion_lines[k+1][j+3+2]),2),round(float(motion_lines[k+1][j+3]),2)]
            #print("START")
            #print(angles_start)
            #print(angles_end)
            #print("END")
            q1 = angles2quat(angles_start[0],angles_start[1],angles_start[2])
            q2 = angles2quat(angles_end[0],angles_end[1],angles_end[2])
            ani_times = numpy.linspace(0, 1, 3)
            p=slerp(q1,q2,ani_times)
            #print(p)
            #print(p[i].xyzw)
            stringa=stringa+str(round(angles_start[2]+p[i].xyzw[2],2))+" "+str(round(angles_start[0]+p[i].xyzw[0],2))+" "+str(round(angles_start[1]+p[i].xyzw[1],2))+" "
        stringa+="\n"
        #print(len(stringa.split()))
        #print(stringa.split())
        #quit()
        #new_bvh.write(stringa)
        #print(stringa.split())
        frames.append(stringa.split())

modifs=[]
d=35
print("APPLYING MEDIAN FITLER")
for i in range(len(frames[0])):
    x=frames[0:len(frames)]
    modif=[]
    for j in range(len(x)):
        #print(x[j][i])
        modif.append(float(x[j][i]))
    #print(modif)
    modif2=signal.medfilt(modif, d)
    for p in range(1):
        x=d
        modif2=signal.medfilt(modif2, x)
    #print(len(modif2))
    modifs.append(modif2)
    #print(modifs[0])
    #for i in range(len(modif)):
        #print(modif[i],modif2[i])
#print("SOMETHING ", len(modifs))
#print(modifs[0])
for i in range(len(modifs[0])):
    print("PROCESSING FRAME "+str(i)+"/"+str(len(modifs[0])))
    stringa=""
    for j in range(len(modifs)):
        stringa+=str(modifs[j][i])+" "
    stringa=stringa+"\n"
    new_bvh.write(stringa) 

print("APPLYING BUTTERWORTH FILTER")
input=output_filename
smooth(input,input.replace(".bvh","_smooth.bvh"))