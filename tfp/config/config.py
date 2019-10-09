import os

DATA_LOC = os.path.join(os.getcwd(),"data") #complete data folder

SPLIT_JSON_LOC = os.path.join(os.getcwd(),"tfp", "config", "split.json")

##file that contain labels generated by parcing cmu website
LABEL_JSON_LOC = os.path.join(os.getcwd(),"tfp", "config", "label.json")

PARENT_LIMBS = {
        21 : [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13],
        15 : [0, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 0, 12,13]
        }

JOINT_NAMES = {
                21:['head_top', 'neck',
                   'right_shoulder', 'right_elbow', 'right_wrist',
                   'left_shoulder', 'left_elbow', 'left_wrist',
                   'right_hip', 'right_knee', 'right_ankle',
                   'left_hip', 'left_knee', 'left_ankle',
                   'pelvis', 'spine', 'head',
                   'right_hand', 'left_hand',
                   'right_foot', 'left_foot',
               ]
              }
LIMB_RATIOS = {
                21 : [2.0, 2.5,
                       1.37, 2.8, 2.4,
                       1.37, 2.8, 2.4,
                       1.05, 4.2, 3.6,
                       1.05, 4.2, 3.6,
                       0, 2.25, 0.8,
                       1.2, 1.2,
                       2.0, 2.0]
            }
ROOT_LIMB = {
            21: 14
        }
HEAD_LIMB = {
            21:0
        }
