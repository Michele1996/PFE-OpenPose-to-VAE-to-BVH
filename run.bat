:donwload all the videos using the ids in the file video_id.txt:
for /F %%i IN (video_test.txt) do youtube-dl -o /videos/%%i.avi https://www.youtube.com/watch?v=%%i:
:compute scene for all the videos:
cd videos/:
for /R %%i in (*) do scenedetect -i %%i -o scenes_out/scenes detect-content -t 27 split-video
cd ..:
python check_static_scene.py
dir videos\scenes\* /b > filelist.txt
:for each scene extract 3d keypoints:
for /F %%i IN (filelist.txt) do bin\OpenPoseDemo.exe --video videos\scenes\%%i  --net_resolution -1x320 --write_json output_final --render_pose 0
python filter.py
:create directory in which the json with keypoints of the same videos are moved
python clean_output_dir