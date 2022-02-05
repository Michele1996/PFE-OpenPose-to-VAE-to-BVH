import numpy


old_bvh=open("out.bvh","r")
new_bvh=open("new_bvh.bvh","w")
l=old_bvh.readlines()
not_to_change=["Hip","Knee","Ankle","Collar"]
lines=[]
index=0
motion_index=-1
motion_lines=[]
for line in l:
    if("JOINT" in line or "ROOT" in line):
        lines.append((line.replace(" ","").replace("JOINT","").replace("ROOT","").replace("\t","").replace("\n",""),index))
        index+=1
        print(lines[index-1])
    if(motion_index>=0):
        motion_lines.append(line.split())
        #print(motion_lines[motion_index])
        motion_index+=1
    if("Frame Time" in line):
        motion_index+=1
#reverse right and left collar, shoulder,elbow and wrist values

for p in range(len(motion_lines)): 
    print("Processing FRAME:"+str(p+1)+"/"+str(len(motion_lines)))
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
                   if("Shoulder" in j[0]):
                       motion_lines[p][i[1]*3+3]=str(float(motion_lines[p][j[1]*3+3]))
                       motion_lines[p][i[1]*3+3+1]=str(float(motion_lines[p][j[1]*3+3+1])+60)
                       motion_lines[p][i[1]*3+3+2]=str(float(motion_lines[p][j[1]*3+3+2])-60)
                       motion_lines[p][j[1]*3+3]=str(float(val_1)-60)
                       motion_lines[p][j[1]*3+3+1]=str(float(val_2)-90)
                       motion_lines[p][j[1]*3+3+2]=str(float(val_3)-30)
                   elif("Elbow" in j[0]):
                       motion_lines[p][i[1]*3+3]=str(float(motion_lines[p][j[1]*3+3])-30)
                       motion_lines[p][i[1]*3+3+1]=str(float(motion_lines[p][j[1]*3+3+1])-45)
                       motion_lines[p][i[1]*3+3+2]=str(float(motion_lines[p][j[1]*3+3+2])-70)
                       motion_lines[p][j[1]*3+3]=str(float(val_1)-45)
                       motion_lines[p][j[1]*3+3+1]=str(float(val_2)+120)
                       motion_lines[p][j[1]*3+3+2]=str(float(val_3))
                   elif("Wrist"in j[0]):
                       motion_lines[p][i[1]*3+3]=str(float(motion_lines[p][j[1]*3+3])-30)
                       motion_lines[p][i[1]*3+3+1]=str(float(motion_lines[p][j[1]*3+3+1])+60)
                       motion_lines[p][i[1]*3+3+2]=str(float(motion_lines[p][j[1]*3+3+2]))
                       motion_lines[p][j[1]*3+3]=str(float(val_1))
                       motion_lines[p][j[1]*3+3+1]=str(float(val_2))
                       motion_lines[p][j[1]*3+3+2]=str(float(val_3))
                   else:
                       motion_lines[p][i[1]*3+3]=str(float(motion_lines[p][j[1]*3+3]))
                       motion_lines[p][i[1]*3+3+1]=str(float(motion_lines[p][j[1]*3+3+1]))
                       motion_lines[p][i[1]*3+3+2]=str(float(motion_lines[p][j[1]*3+3+2]))
                       motion_lines[p][j[1]*3+3]=str(float(val_1))
                       motion_lines[p][j[1]*3+3+1]=str(float(val_2))
                       motion_lines[p][j[1]*3+3+2]=str(float(val_3))
       
for line in l:
    if("Frame Time" in line):
        new_bvh.write(line)
        break
    else:
        new_bvh.write(line)

for k in motion_lines:
    stringa=""
    for j in k:
        stringa+=str(j)+" "
    stringa+="\n"
    new_bvh.write(stringa)