import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import BridgeDataRailRLPrivateVRWidowX
from widowx_envs.control_loops import TimedLoop
from widowx_envs.policies.vr_teleop_policy import VRTeleopPolicy
from rlkit.torch.sac.policies.resnet34_gaussian_policy import GaussianResNetPolicy
import torch

# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_carrot_on_cutting_board/2021-06-08_18-42-42/raw/traj_group0/traj0', 0]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen1/put_sweet_potato_in_pot_which_is_in_sink_distractors/2021-06-03_16-50-31/raw/traj_group0/traj19', 0]
load_traj = ['/home/dcuser1/trainingdata/robonetv2/toykitchen_fixed_cam_initial_testconfig//2021-09-15_16-30-18/raw/traj_group0/traj0', 100]


env_params = {
    'camera_topics': [IMTopic('/cam0/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    # 'action_mode':'3trans3rot',
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'override_workspace_boundaries': [[0.2, -0.04, 0.03, -1.57, 0], [0.31, 0.04, 0.1,  1.57, 0]],
    'action_clipping': None,
    'start_transform': load_traj,
}

cnn_params=dict(
    input_width=128,
    input_height=128,
    input_channels=3,
    hidden_sizes=[1024, 512, 256],
)

# reward_predictor = GaussianResNetPolicy(
#     max_log_std=0,
#     min_log_std=-6,
#     obs_dim=None,
#     action_dim=1,
#     std=0.01,
#     output_activation='sigmoid',
#     added_fc_input_size=54,
#     **cnn_params,
# ).cuda()


reward_predictor = torch.load(
    '/home/dcuser1/robonetv2bucket/spt_data/experiments/railrl_experiments/take-out-broc-0.03hue/take_out_broc_0.03hue_2021_11_04_18_34_24_0000--s-0/reward_predictor_itr_50.pt'
    # '/home/dcuser1/robonetv2bucket/spt_data/experiments/railrl_experiments/rew-put-broc-pot/rew_put_broc_pot_2021_11_04_18_38_14_0000--s-0/reward_predictor_itr_80.pt'
)['evaluation/reward_predictor']
# import ipdb; ipdb.set_trace()
reward_predictor.eval()

task_id = 0
num_tasks = 1

agent = {
    'type': TimedLoop,
    'env': (BridgeDataRailRLPrivateVRWidowX, env_params, reward_predictor, task_id, num_tasks),
    'T': 500,
    'image_height': 480,  # for highres
    'image_width': 640,   # for highres
    'make_final_gif': True,
    'recreate_env': (False, 1),
    'ask_confirmation': False,
}

policy = {
    'type': VRTeleopPolicy,
    'log': True,
    # 'path': '/home/dcuser1/experiments/railrl_experiments/kitchen1-bc/kitchen1_bc_2021_08_05_01_34_32_0000--s-0/model_pkl/290.pt',
    # 'path': '/home/dcuser1/experiments/railrl_experiments/awac-resnet-all/awac_resnet_all_2021_09_10_09_13_51_0000--s-0/model_pkl/160.pt',
    # 'exp_conf_path': '/home/dcuser1/experiments/railrl_experiments/kitchen1-bc/kitchen1_bc_2021_08_05_01_34_32_0000--s-0/variant.json',
    # 'exp_conf_path': '/home/dcuser1/experiments/railrl_experiments/awac-resnet-all/awac_resnet_all_2021_09_10_09_13_51_0000--s-0/variant.json',
    'num_tasks': 54,
    'task_id': 9,
    'confirm_first_image': True,
    'resnet': True,
    'normalize': False,
    'load_qfunc': False,
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
