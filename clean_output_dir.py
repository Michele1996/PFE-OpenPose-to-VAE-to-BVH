import os
path1 = os.getcwd()+"\\output"
files = os.listdir(path)
for file in files:
       if(not os.path.isdir(path+file.split("-")[0])):
         os.mkdir(path1+file.split("-")[0])
       print("Processing keypoints of file "+file+" , move to directory "+ path+file.split("-")[0])
       os.rename(os.path.join(path, file), os.path.join(path+file.split("-")[0],file))