import json
import glob

def repair_metadata():

    dated_folders = glob.glob('/mount/harddrive/trainingdata/robonetv2/toykitchen_fixed_cam/upenn/toykitchen3/turn_faucet_left_56/*')
    for folder in dated_folders:
        try:
            metadata = json.load(open(folder + '/collection_metadata.json', 'r'))
        except:
            print('failed loading', folder)
            continue

        metadata = {
            "environment": "toykitchen3",
            "camera_configuration": "table_top",
            "camera_type": "Realsense D455",
            "policy_desc": "human_demo, turn faucet left",
            "robot": "widowx",
            "gripper": "default",
            "background": "table",
            "action_space": "x,y,z,roll, pitch, yaw, grasp_continuous",
            "object_classes": "faucet, distractors"
        }
        # import pdb; pdb.set_trace()

        with open(folder + '/collection_metadata.json', 'w') as f:
            print('writing ', folder)
            json.dump(metadata, f, indent=4)




if __name__ == '__main__':
    repair_metadata()