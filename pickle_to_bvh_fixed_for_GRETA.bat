python save_data_json_openpose_format
move Test_Droite ./MocapNET-master/Test_Droite
move Test_Gauche ./MocapNET-master/Test_Gauche
cd MocapNET-master
del droite.bvh
del gauche.bvh
:a utiliser avec bash linux:
wsl ./convertOpenPoseJSONToCSV --from Test_Droite --size 480 480:
wsl ./MocapNET2CSV --from Test_Droite/2dJoints_v1.4.csv --size 480 480 --novisualization
ren out.bvh droite.bvh
wsl ./convertOpenPoseJSONToCSV --from Test_Gauche --size 480 480:
wsl ./MocapNET2CSV --from Test_Gauche/2dJoints_v1.4.csv --size 480 480 --novisualization
ren out.bvh gauche.bvh
cd ..
python fix_bvh_for_GRETA_test_mir_fusion_gauche_droite  gauche.bvh droit.bvhe