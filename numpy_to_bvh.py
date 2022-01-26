import numpy
dictionary_skeleton={"hip":0,"rButtock":1,"rThigh":2,"rShin":3,"lButtock":4,"lThigh":5,"lShin":6,"abdomen":7,"chest":8,"neck":9,"head":10,"lShldr":11,"lForeArm":12,"lHand":13,"rShldr":14,"rForeArm":15,"rHand":16}
dictionary_skeleton_2={"hip":0,"rThigh":1,"rShin":2,"rFoot":3,"lThigh":4,"lShin":5,"lFoot":6,"abdomen":7,"chest":8,"neck":9,"head":10,"lShldr":11,"lForeArm":12,"lHand":13,"rShldr":14,"rForeArm":15,"rHand":16}
dictionary_skeleton_3={"hip":0,"lButtock":1,"lThigh":2,"lShin":3,"rButtock":4,"rThigh":5,"rShin":6,"abdomen":7,"chest":8,"neck":9,"head":10,"rCollar":11,"rShldr":12,"rForeArm":13,"lCollar":14,"lShldr":15,"lForeArm":16}
x=numpy.load("preds3D.npy")
diction={i : x[i] for i in  range(len(x))}
#print(dictionary_skeleton)
bvh = open("out.bvh", "r")
new_bvh=open("new.bvh","w")
p_bvh=bvh.readlines()
lines=[]
for line in p_bvh:
    if("JOINT" in line or "ROOT" in line):
        lines.append(line.replace(" ","").replace("JOINT","").replace("ROOT","").replace("\t",""))
dictionary_bvh={i:lines[i].replace("\n","") for i in range(len(lines))}
for line in p_bvh:
    new_bvh.write(line)
    if("MOTION" in line):
        break
new_bvh.write("Frames: "+str(len(x))+"\n")
new_bvh.write("Frame Time: 0.04\n")
new_line=[]
index=0
num_val=3
correct_skel_added=0
for i in range(len(x)):
    index=0
    print("PROCESSING "+str(i+1)+"/"+str(len(x))+" FRAME")
    for p in range(len(lines)):

        if(dictionary_bvh[p] in dictionary_skeleton_2.keys()):
           correct_skel_added+=1
           val=diction[i][dictionary_skeleton_2[dictionary_bvh[p]]]
           new_line.append(val[0]*100)
           new_line.append(val[1]*100)
           new_line.append(val[2]*100)
           if(dictionary_bvh[p]=="hip"):
              print("hips")
              new_line.append(0)
              new_line.append(0)
              new_line.append(0)
        else:
            for k in range(3):
                new_line.append(0)
    string=""
    for z in new_line:
        string=string+str(z)+" "
    string=string+"\n"
    string=string.replace("[","").replace("]","")
    print(len(string.split(" ")), correct_skel_added)
    new_bvh.write(string)
    new_line=[]
    correct_skel_added=0
p=open("new.txt","r")
d=p.read()
print(len(d.split(" ")))
print(d.split(" "))