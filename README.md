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
(NOTE you will need to install youtube_dl https://pypi.org/project/youtube_dl/, pyscenedetect and ffmpeg https://pyscenedetect.readthedocs.io/en/latest/download/, and OpenPose https://github.com/CMU-Perceptual-Computing-Lab/openpose , using Body25 keypoints detection)

## Variational Auto-Encoder and Data Generation

Now we will need to create a .npy file using the json output file of OpenPose.\\
You can create it using the following command
```python
python create_npy
```
Now you can execute 
```python
python var_autoencoder nb_iter nb_frames
```
To run the VAE and to generate data (and also visualize. if you need check the vae_autoencoder script) and save them in the same format as OpenPose output (we use all the 25 keypoints) \\
The output of OpenPose is a json that contains among other information the x,y,confiance_score data for each of the 25 keypoints. We need to add to the encoder generated data, the confiance_score (just to make it work with MocapNET) so we can put it to 1 for all the keypoints
