
# General:
- vary kitchen position, and camera every 25
- vary distractors every 5 (it's enough to move about 10% of distractors every time)
- update the metadata *before* every new round of data collection, to do that, edit `widowx_envs/experiments/toykitchen_fixed_cam/collection_metadata.json`

## Example metadata:
{
    "environment": "toykitchen1",
    "camera_configuration": "table_top",        # what are the cameras looking at? Other examples are 'fridge', 'microwave'
    "robot_base_position_variation": "12-3-40",  # change this every time you change the robot base position, putting the date here plus a counter should be enough
    "camera_variation": "12-3-40",  # change this every time you change the camera positions
    "camera_type": "Logitech C920",
    "policy_desc": "human demo",
    "robot": "widowx",
    "gripper": "default",
    "action_space": "x,y,z,roll, pitch, yaw, grasp_continuous",
    "object_classes": "pepper shaker, distractors"  
}


# commands

## start robot and cameras:
    ~/interbotix_ws/src/robonetv2$ bash scripts/run.sh -c /home/dcuser1/interbotix_ws/src/robonetv2/widowx_envs/widowx_envs/utils/multicam_server_rospkg/src/usb_connector_chart_user.yml

## visualize cameras:
(from anywhere)
$ rviz
then select File > recent configs > rivz_6cam_config.rviz

## collection script:
~/interbotix_ws/src/robonetv2/widowx_envs$ python widowx_envs/run_data_collection.py experiments/toykitchen_fixed_cam_human/conf.py --prefix toykitchen1/open_microwave

the prefix argument starts with the environment e.g. toykitchen1 and separated by '/' contains the task name

## Verification:
## always verify data collection result
1. copy the trajectory path from the terminal ctrl+shift+c
2. open file browser
3. in the file browser ctrl+l and past the path
4. check the images