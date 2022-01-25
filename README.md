# PFE-OpenPose-to-VAE-toBVH

The aim of the project is to generate gestures using VAE trained using keypoints acquired using OpenPose on TEDx videos thata were splitted in scenes
The project is split in 3 phases:
* Generate de data
  * Use youtube_dl to download the videos
  * Use then pyscenedetect to split them in scenes
  * Filter the scenes (statique scenes)
* Apply OpenPose on the scenes
* Filter the scenes (back camera, body not well oriented, body parts missing....)
* Use the outputs to train a VAE
* Generate Data using the encoder
* Reconstruct json file as OpenPose output and use MocapNET to have bvh files 
