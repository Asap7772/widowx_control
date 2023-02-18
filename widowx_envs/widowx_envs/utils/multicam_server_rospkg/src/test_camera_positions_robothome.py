import rospy
import cv2

import numpy as np

import matplotlib.pyplot as plt
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.utils.multicam_server_rospkg.src.camera_recorder import CameraRecorder
from widowx_envs.widowx.widowx_env import WidowXEnv
import os
import pickle as pkl

tstep = 11
traj_path = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_potato_on_plate/2021-07-05_15-56-49/raw/traj_group0/traj20'
load_traj = [traj_path, tstep]

if __name__ == '__main__':

    camera_topic = IMTopic('/cam0/image_raw')

    env_params = {
    'camera_topics': [camera_topic],
    'gripper_attached': 'custom',
    'start_transform': load_traj,
    'move_to_rand_start_freq':-1
    }

    env = WidowXEnv(env_params)

    env.move_to_startstate()

    rec = CameraRecorder(camera_topic)
    rospy.sleep(1)
    reference_image = '/home/datacol1/Desktop/reference_image.jpg'
    reference_image = cv2.imread(reference_image)
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    # tstamp, im = rec.get_image()
    # cv2.imwrite('/home/datacol1/Desktop/reference_image.jpg', im)

    print('press esc to quit!')
    while not rospy.is_shutdown():
        tstamp, im = rec.get_image()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        averaged = ((im.astype(np.float32) + reference_image.astype(np.float32))/2).astype(np.uint8)
        cv2.imshow('image', averaged[:, :, ::-1])
        if cv2.waitKey(1) == 27:
            break  # esc to quit
