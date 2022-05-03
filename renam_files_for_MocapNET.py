import os
import shutil
import argparse
CRED = '\033[91m'
CEND = '\033[0m'
parser = argparse.ArgumentParser()
parser.add_argument("directory", help="directory in which there are the json to rename",type=str,nargs='?', default="\\output\\0")
args = parser.parse_args()
or_path=os.getcwd()
path = os.getcwd()+args.directory
files = os.listdir(path)
i=0
if(not os.path.isdir(os.getcwd()+"\\MocapNET-master")):
    print("Error, MocapNET not found\nPlease install MocapNET https://github.com/FORTH-ModelBasedTracker/MocapNET")
    quit()
if(not os.path.isdir(os.getcwd()+"\\MocapNET-master\\BRUoutPATS_FEMME")):
    os.mkdir(os.getcwd()+"\\MocapNET-master\\BRUout1")
for file in files:
       index="0000"+str(i)
       decalage=len(index)-5
       index=index[decalage:len(index)]
       shutil.copyfileobj
       file_src = path+"\\"+file  
       f_src = open(file_src, 'rb')
       file_dest =or_path+"\\MocapNET-master\\BRUoutPATS_FEMME\\"+"color_"+file[0]+"_"+index+"_keypoints.json"  
       f_dest = open(file_dest, 'wb')
       shutil.copyfileobj(f_src, f_dest) 
       #os.rename(os.path.join(path, file), os.path.join(or_path+"\\MocapNET-master\\BRUout","color_"+file[0]+"_"+index+"_keypoints.json"))
       i+=1

