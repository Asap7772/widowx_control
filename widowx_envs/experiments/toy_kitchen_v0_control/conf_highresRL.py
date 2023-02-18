import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop
from semiparametrictransfer.policies.rl_policy_cog import RLPolicyCOG

load_traj = [os.environ['DATA'] + '/robonetv2/demo_data/berkeley/toykitchen1/take_broccoli_out_of_pan/2021-07-16_17-10-32/raw/traj_group0/traj0', 6]

env_params = {
    'camera_topics': [IMTopic('/cam0/image_raw')],
    # 'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    # 'action_mode':'3trans3rot',
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'override_workspace_boundaries': [[0.2, -0.04, 0.03, -1.57, 0], [0.31, 0.04, 0.1,  1.57, 0]],
    'action_clipping': None,
    'start_transform': load_traj,
}

agent = {
    'type': TimedLoop,
    'env': (WidowXEnv, env_params),
    'T': 50,
    'image_height': 480,  # for highres
    'image_width': 640,   # for highres
    'make_final_gif': True,
    # 'video_format': 'gif',   # already by default
    'recreate_env': (False, 1),
    'ask_confirmation': False,
    # 'load_goal_image': [load_traj, 18],
}

policy = {
    'type': RLPolicyCOG,
    'log': True,
    # 'path' : '/home/dcuser1/anikait/bottleneck1e-1-task1-minq1/110.pt',
    # 'path' : '/home/dcuser1/anikait/exps_nov19/everystate_minqv3_minq1/250.pt',
    'path': '/home/dcuser1/anikait/exps_nov19/normconv_minqv2_minq1/110.pt',    
    # 'policy_type': 1,
    'history': True,
    'history_size': 2,
    # 'optimize_q_function': True,
    'bottleneck': True,
    # 'vqvae': False,
}

config = {
    # 'collection_metadata' : current_dir + '/collection_metadata.json',
    'current_dir': current_dir,
    'start_index': 0,
    'end_index': 300,
    'agent': agent,
    'policy': policy,
    'save_data': True,  # by default
    'save_format': ['raw'],
}