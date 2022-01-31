python var_autoencoder.py 3 20
cd MocapNET-master
wsl.exe ./convertOpenPoseJSONToCSV --from OUTPUT_to_BVH --size 640 480
wsl.exe ./MocapNET2CSV --from BRUout/2dJoints_v1.4.csv --size 640 480 --novisualization