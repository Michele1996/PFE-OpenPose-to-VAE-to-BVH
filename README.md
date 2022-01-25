# PFE-OpenPose-to-VAE-toBVH

The aim of the project is to generate gestures using VAE trained using keypoints acquired using OpenPose on TEDx videos thata were splitted in scenes
The project is split in 3 phases:
* Generate de data from TEDx videos
  * Use youtube_dl to download the videos
  * Use then pyscenedetect to split them in scenes
  * Filter the scenes (statique scenes)
* Apply OpenPose on the scenes
* Filter the scenes (back camera, body not well oriented, body parts missing....)
* Use the outputs to train a VAE
* Generate Data using the encoder
* Reconstruct json file as OpenPose output and use MocapNET to have bvh files 

## Generate Data
First of all go to the directory and run the batch file 
```batch
run.bat
```
This file will download the videos (in video_ids.txt), split them in scenes, filter the scene , apply OpenPose and then filter the keypoints output json file
(NOTE you will need to install youtube_dl https://pypi.org/project/youtube_dl/, pyscenedetect and ffmpeg https://pyscenedetect.readthedocs.io/en/latest/download/, and OpenPose https://github.com/CMU-Perceptual-Computing-Lab/openpose)
