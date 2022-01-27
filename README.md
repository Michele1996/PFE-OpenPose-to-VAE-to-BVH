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

Now we will need to create a .npy file using the json output file of OpenPose.<br />
You can create it using the following command
```python
python create_npy
```
Now you can execute 
```python
python var_autoencoder nb_iter nb_frames
```
Where nb_iter is the number of itereration creation of nb_frames frames by encoder <br />
To run the VAE and to generate data (and also visualize. if you need check the vae_autoencoder script) and save them in the same format as OpenPose output (we use all the 25 keypoints) <br />
The output of OpenPose is a json that contains among other information the x,y,confiance_score data for each of the 25 keypoints. We need to add to the encoder generated data, the confiance_score (just to make it work with MocapNET) so we can put it to 1 for all the keypoints

## Generate BVH from json using MocapNET
You will need to clone https://github.com/FORTH-ModelBasedTracker/MocapNET in the directory and then follow their guide to build it 
Now you can execute using a bash linux the following command:
```python
./convertOpenPoseJSONToCSV --from OUTPUT_to_BVH --size 640 480
./MocapNET2CSV --from OUTPUT_to_BVH/2dJoints_v1.4.csv --size 640 480 --novisualization:
```
Where OUTPUT_to_BVH is the output inside MocapNET which contains all the json file created as shown below. All the json files names need to follow the pattern colorFrame_nb_video_XXXXX_keypoints.json.
<br />For example, you will have in the directory colorFrame_0_00000_keypoints.json, colorFrame_0_00001_keypoints.json etc.
<br />
You can also run 
```batch
from_vae_to_json_to_bvh.bat
```
which will run all the process from train the autoencoder to create the bvh (it uses wsl linux bash)
<br />
You can check https://github.com/FORTH-ModelBasedTracker/MocapNET/blob/master/doc/OpenPose.md if you have any doubt about the conversion made by MocapNET using the OpenPose json output

## Test on Blender
Now you have from a 2d Keypoints data , a 3D animation. You can check on Blender by importing the bvh file , the result of your animation
(To add a human model to the skeleton create by the bvh file you can check https://www.youtube.com/watch?v=GBSC10euloY)
<br /><br />
You can also visualize the animation on http://lo-th.github.io/olympe/BVH_player.html which is a visualization-only tool

## Conversion of Numpy Array to BVH
In case of you have a numpy array with shape (nb_frames,nb_keypoints,3) you can use the script
```python
python numpy_to_bvh
```
to convert your numpy_array in a bvh animation. <br/>
Check in the script the structure of the skeleton and modify it if you need. You can add a human model on the skeleton following the youtube tutorial above

