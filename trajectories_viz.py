import pybullet as p
import pybullet_data
import pandas as pd
import numpy as np
import time as t

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
data_list = []
for i in range(1,5):
    trajectory_path = 'bipedal_env\\trajectories\\motion_ful_0'+str(i)+'.csv'
    data_list += [pd.read_csv(trajectory_path)]
data = pd.concat(data_list,
                 axis = 0,
                 join="outer",
                ignore_index=False,
                keys=None,
                levels=None,
                names=None,
                verify_integrity=False,
                copy=True,) 
# print(data.head())
# data.to_csv('bipedal_env\\trajectories\\motion_full.csv',index=False)
num_points = data.shape[1]//3
point_list = []
for i in range(num_points):
    point_list.append(p.loadURDF('bipedal_env\\target.urdf'))
for i in range(data.shape[0]):
    # point_cordinate = []
    for point in point_list:
        base_cordinate = np.array((data.iloc[i,2]/20,data.iloc[i,0]/15,0))
        point_cordinate = np.array((data.iloc[i,3*point+2]/20,data.iloc[i,3*point]/15,data.iloc[i,3*point+1]/20))
        cordinate = point_cordinate-base_cordinate
        data.iloc[i,3*point] = cordinate[0]
        data.iloc[i,3*point+1] = cordinate[1]
        data.iloc[i,3*point+2] = cordinate[2]
        p.resetBasePositionAndOrientation(point,cordinate,[0,0,0,1])
    # p.addUserDebugPoints(point_cordinate,color_list,pointSize = 4., lifeTime = .0001)
    # t.sleep(.05)
    p.stepSimulation()
print(data.head())
# data.to_csv('bipedal_env\\trajectories\\motion_cordinate.csv',index=False)
lthigh_vec = data.iloc[:,:3].to_numpy() - data.iloc[:,3:6].to_numpy()
rthigh_vec = data.iloc[:,:3].to_numpy() - data.iloc[:,9:12].to_numpy()
l_bicep = data.iloc[:,3:6].to_numpy() - data.iloc[:,6:9].to_numpy()
r_bicep = data.iloc[:,9:12].to_numpy() - data.iloc[:,12:15].to_numpy()
print(lthigh_vec)