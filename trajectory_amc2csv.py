import amc_parser as amc
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
joints_path = 'bipedal_env\\trajectories\\09.asf'
motion_path = 'bipedal_env\\trajectories\\09_04.amc'
joints = amc.parse_asf(joints_path)
motions = amc.parse_amc(motion_path)
def get_joint_pos_dict(c_joints, c_motion):
    c_joints['root'].set_motion(c_motion)
    out_dict = {}
    for k1, v1 in c_joints['root'].to_dict().items():
        for k2, v2 in zip('xyz', v1.coordinate[:, 0]):
            out_dict['{}_{}'.format(k1, k2)] = v2
    return out_dict
motion_df = pd.DataFrame([get_joint_pos_dict(joints, c_motion) for c_motion in motions])
motion_df.to_csv('bipedal_env\\trajectories\\motion_full_04.csv', index=False)