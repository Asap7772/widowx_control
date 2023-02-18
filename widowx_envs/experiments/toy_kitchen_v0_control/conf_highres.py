import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop

# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/tool_chest/pick_up_blue_pen_and_put_into_drawer/2021-07-23_19-23-00/raw/traj_group0/traj23', 0]
load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/2021-09-13_23-34-45/raw/traj_group0/traj0', 65]


env_params = {
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'override_workspace_boundaries': [[0.2, -0.04, 0.03, -1.57, 0], [0.31, 0.04, 0.1,  1.57, 0]],
    'action_clipping': None,
    'start_transform': load_traj,
}

agent = {
    'type': TimedLoop,
    'env': (WidowXEnv, env_params),
    'T': 40,
    'image_height': 480,  # for highres
    'image_width': 640,   # for highres
    'make_final_gif': False,
    'recreate_env': (False, 1),
    'ask_confirmation': False,
}

policy = [
{
    'type': GCBCPolicyImages,

    # kitchen 1 random mix:
    # "human_demo, pick up box cutter and put into drawer": 0,
    # "human_demo, put fork from basket to tray": 1,
    # "human_demo, put lid on stove": 2,
    # "human_demo, flip pot upright which is in sink": 3,
    # "human_demo, take sushi out of pan": 4,
    # "human_demo, put eggplant into pot or pan": 5,
    # "human_demo, put cup into pot or pan": 6,
    # "human_demo, put potato in pot or pan": 7,
    # "human_demo, take lid off pot or pan": 8,
    # "human_demo, put big spoon from basket to tray": 9,
    # "human_demo, put carrot in pot or pan": 10,
    # "human_demo, put corn in pan which is on stove": 11,
    # "human_demo, pick up violet Allen key": 12,
    # "human_demo, take can out of pan": 13,
    # "human_demo, put spatula in pan": 14, # TODO
    # "human_demo, put brush into pot or pan": 15, # TODO
    # "human_demo, put spoon in pot": 16,
    # "human_demo, put pear in bowl": 17,
    # "human_demo, put knife in pot or pan": 18,
    # "human_demo, put detergent from sink into drying rack": 19,
    # "human_demo, put can in pot": 20,
    # "human_demo, put carrot on cutting board": 21,
    # "human_demo, put banana on plate": 22,
    # "human_demo, lift bowl": 23,
    # "human_demo, twist knob start vertical _clockwise90": 24,
    # "human_demo, put lid on pot or pan": 25,
    # "human_demo, pick up red screwdriver": 26,
    # "human_demo, put pot or pan in sink": 27,
    # "human_demo, pick up pot from sink": 28,
    # "human_demo, put pepper in pot or pan": 29,
    # "human_demo, pick up pan from stove": 30,
    # "human_demo, put strawberry in pot": 31, # TODO
    # "human_demo, put broccoli in bowl": 32,
    # "human_demo, put potato on plate": 33, # TODO
    # "human_demo, put small spoon from basket to tray": 34,
    # "human_demo, put pepper in pan": 35,
    # "human_demo, put eggplant on plate": 36,
    # "human_demo, pick up scissors and put into drawer": 37,
    # "human_demo, put pot or pan from sink into drying rack": 38,
    # "human_demo, put corn into bowl": 39,
    # "human_demo, flip cup upright": 40,
    # "human_demo, pick up glue and put into drawer": 41,
    # "human_demo, pick up closest rainbow Allen key set": 42,
    # "human_demo, put sweet potato in pot": 43,
    # "human_demo, put spoon into pan": 44,
    # "human_demo, take carrot off plate": 45,
    # "human_demo, flip orange pot upright in sink": 46,
    # "human_demo, take broccoli out of pan": 47,
    # "human_demo, flip salt upright": 48,
    # "human_demo, put sushi on plate": 49,
    # "human_demo, put cup from anywhere into sink": 50,
    # "human_demo, turn faucet front to left (in the eyes of the robot)": 51,
    # "human_demo, put green squash into pot or pan": 52,
    # "human_demo, put broccoli in pot or pan": 53,
    # "human_demo, put sweet_potato in pan which is on stove": 54,
    # "human_demo, put carrot on plate": 55,
    # "human_demo, turn lever vertical to front": 56,
    # "human_demo, put corn in pot which is in sink": 57,
    # "human_demo, pick up bit holder": 58,
    # "human_demo, pick up blue pen and put into drawer": 59,
    # "human_demo, put knife on cutting board": 60,
    # "human_demo, put red bottle in sink": 61,
    # "human_demo, put pot or pan on stove": 62,
    # "human_demo, put corn on plate": 63,
    # "human_demo, put lemon on plate": 64,
    # "human_demo, put detergent in sink": 65

    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/random_mixing_task_id/random_mixing_kitchen1_2021-08-15_20-04-45/weights/weights_best_itr119510.pth',

    # "human_demo, put carrot in bowl": 1,
    # "human_demo, put banana in pot or pan": 2,
    # "human_demo, turn lever vertical to front": 3,
    # "human_demo, put pear on plate": 4,
    # "human_demo, put can in pot": 5,
    # "human_demo, flip salt upright": 6,
    # "human_demo, put carrot in pot or pan": 7,
    # "human_demo, put carrot on plate": 8,
    # "human_demo, put potato on plate": 9,
    # "human_demo, put cup into pot or pan": 10,
    # "human_demo, put knife in pot or pan": 11,
    # "human_demo, put detergent from sink into drying rack": 12,
    # "human_demo, put eggplant into pot or pan": 13,
    # "human_demo, pick up sponge and wipe plate": 14,
    # "human_demo, put lemon on plate": 15,
    # "human_demo, put spatula in pan": 16,
    # "human_demo, put spoon into pan": 17,
    # "human_demo, put brush into pot or pan": 18,
    # "human_demo, put cup from anywhere into sink": 19,
    # "human_demo, flip orange pot upright in sink": 20,
    # "human_demo, put potato in pot or pan": 21,
    # "human_demo, flip pot upright which is in sink": 22,
    # "human_demo, flip cup upright": 23,
    # "human_demo, put corn on plate": 24,
    # "human_demo, open box": 25,
    # "human_demo, put sushi in pot or pan": 26,
    # "human_demo, close box": 27,
    # "human_demo, put pot or pan in sink": 28,
    # "human_demo, put sweet potato in pot": 29,
    # "human_demo, put pot or pan on stove": 30,
    # "human_demo, pick up green mug": 31,
    # "human_demo, put knife on cutting board": 32,
    # "human_demo, take lid off pot or pan": 33,
    # "human_demo, pick up any cup": 34,
    # "human_demo, put detergent in sink": 35,
    # "human_demo, pick up glass cup": 36,
    # "human_demo, put green squash into pot or pan": 37,
    # "human_demo, put sushi on plate": 38,
    # "human_demo, lift bowl": 39,
    # "human_demo, put lid on pot or pan": 40,
    # "human_demo, put pear in bowl": 41,
    # "human_demo, put pot or pan from sink into drying rack": 42,
    # "human_demo, put spoon in pot": 43,
    # "human_demo, put strawberry in pot": 44,

    'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/task_id_conditioned/task_id_excl_kitchen1_2021-09-10_09-16-47/weights/weights_best_itr75720.pth',

    'confirm_first_image': True,
    # 'crop_image_region': [31, 88],
    'model_override_params': {
        'data_conf': {
            'random_crop': [96, 128],
            'image_size_beforecrop': [112, 144]
        },
        'img_sz': [96, 128],
        'sel_camera': 0,
        'state_dim': 7,
        'test_time_task_id': 21,
    },
}
]


config = {
    # 'collection_metadata' : current_dir + '/collection_metadata.json',
    'current_dir': current_dir,
    'start_index': 0,
    'end_index': 300,
    'agent': agent,
    'policy': policy,
    'save_data': True,
    'save_format': ['raw'],
}