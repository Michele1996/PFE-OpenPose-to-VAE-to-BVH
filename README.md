# PFE-OpenPose-to-VAE-to-BVH

The aim of the project is to generate gestures using VAE trained using keypoints acquired using OpenPose on TEDx videos thata were splitted in scenes
The project is split in 3 phases:
* Generate de data from TEDx videos
  * Use youtube_dl to download the videos
  * Use then pyscenedetect to split them in scenes
  * Filter the scenes (statique scenes)
* Apply OpenPose on the scenes
* Filter the keypoints (back camera, body not well oriented, body parts missing....)
* Use the outputs to train a VAE
* Generate Data using the encoder
* Reconstruct json file as OpenPose output and use MocapNET to have bvh files 

<p align="center">
  <img src="https://github.com/Michele1996/PFE-OpenPose-to-VAE-to-BVH/blob/main/images/compare.JPG" width="800" height="600" alt="Schema real samples and reconstructions"/>
</p>
<p align="center">
  <img src="https://github.com/Michele1996/PFE-OpenPose-to-VAE-to-BVH/blob/main/images/motion_bvh.JPG" alt="Schema real samples and reconstructions"/>
</p>

## Generate Data
First of all go to the directory and run the batch file
```batch
run.bat
```
This file will download the videos (in video_ids.txt), split them in scenes, filter the scene , apply OpenPose and then filter the keypoints output json file
(NOTE you will need to install youtube_dl https://pypi.org/project/youtube_dl/, pyscenedetect and ffmpeg https://pyscenedetect.readthedocs.io/en/latest/download/, and OpenPose https://github.com/CMU-Perceptual-Computing-Lab/openpose , using Body25 keypoints detection)

## Filtering Scripts (Scenes and Keypoints)
If you execute
```python
python check_static_scene.py
```
All the scenes that are static(as an image showed in a video, or a power point presentation and so on) will be deleted . The algorithm compute the difference  intensity in intensity between each frame and if the difference is  lower than a threshold , it is added to a list of frame that are similar. If more than 20% of the frames are similar the scene is deleted.<br />
While by executing:
```python
python filter.py
```
You will filter the keypoints. More precisely all the OpenPose json output will be deleted if any of the body part is missig, if the camera is on the back of the speaker, if the skeleton is not big enough, if the json contains data for more than 1 person or does not contains data (nobody in the scene)<br/>
You can find keypoints filtered for 150 videos at the following link: https://drive.google.com/drive/folders/1iM7aa9yEMOSrr4X3Q7yvBCYomg9xdY3i?usp=sharing

## Variational Auto-Encoder and Data Generation

Now we will need to create a .npy file fro one video using the json output file of OpenPose.<br />
You can create it using the following command
```python
python create_npy_one_video.py
```
Or you can use this other script to create a .npy which will include all the json for all the videos
```python
python create_npy_all_video.py
```
Now you can execute 
```python
python var_autoencoder nb_iter nb_frames save_test path_to_data
```
Where nb_iter is the number of itereration creation of nb_frames frames by encoder , save_test is a boolean and if is True, than even if you don't have MocapNET installed the generated data will be saved int Test/OUTPUT_to_BVH anyway, and path_to_data is the path to .npy file containing the OpenPose output. (you can avoid give as argument save_test and path_to_data, they are set by default on False, and path to keypoints_150_videos.npy <br /><br/>
To run the VAE and to generate data (and also visualize. if you need check the vae_autoencoder script) and save them in the same format as OpenPose output (we use all the 25 keypoints) <br />
The output of OpenPose is a json that contains among other information the x,y,confiance_score data for each of the 25 keypoints. We need to add to the encoder generated data, the confiance_score (just to make it work with MocapNET) so we can put it to 1 for all the keypoints

## Generate BVH from json using MocapNET
You will need to clone https://github.com/FORTH-ModelBasedTracker/MocapNET in the directory and then follow their guide to build it 
Now you can execute using a bash linux the following command:
```python
./convertOpenPoseJSONToCSV --from OUTPUT_to_BVH --size 640 480
./MocapNET2CSV --from OUTPUT_to_BVH/2dJoints_v1.4.csv --size 640 480 --novisualization
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
In case of you have a numpy array with shape (nb_frames,nb_keypoints,3) you can use the following script to convert your numpy_array in a bvh animation
```python
python numpy_to_bvh
```
Check in the script the structure of the skeleton and modify it if you need. You can add a human model on the skeleton following the youtube tutorial above

