import os
path = os.getcwd()+"\\output_final"
files = os.listdir(path)
i=0
os.mkdir(path+"\\BRUout")
for file in files:
       name=file[0]
       if(name=="0"):
           index="0000"+str(i)
           decalage=len(index)-5
           index=index[decalage:len(index)]
           os.rename(os.path.join(path, file), os.path.join(path+"\\BRUout","color_"+file[0]+"_"+index+"_keypoints.json"))
           i+=1