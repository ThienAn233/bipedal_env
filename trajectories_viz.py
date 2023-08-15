import pybullet as p
import pybullet_data
import pandas as pd
import numpy as np
import time as t

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
data_list = []
# for i in range(1,5):
#     trajectory_path = 'bipedal_env\\trajectories\\motion_full_0'+str(i)+'.csv'
#     data_list += [pd.read_csv(trajectory_path)]
# data = pd.concat(data_list,
#                  axis = 0,
#                  join="outer",
#                 ignore_index=False,
#                 keys=None,
#                 levels=None,
#                 names=None,
#                 verify_integrity=False,
#                 copy=True,) 
data = pd.read_csv('bipedal_env\\trajectories\\motion_full_04.csv')
# print(data.head())
# data.to_csv('bipedal_env\\trajectories\\motion_full.csv',index=False)
num_points = data.shape[1]//3
point_list = []
new_data = data.copy()
for i in range(num_points):
    point_list.append(p.loadURDF('bipedal_env\\target.urdf'))
for i in range(data.shape[0]):
    # point_cordinate = []
    for point in point_list:
        base_cordinate = np.array((data.iloc[i,2]/20,data.iloc[i,0]/15,0))
        point_cordinate = np.array((data.iloc[i,3*point+2]/20,data.iloc[i,3*point]/15,data.iloc[i,3*point+1]/20))
        cordinate = point_cordinate-base_cordinate
        new_data.iloc[i,3*point] = cordinate[0]
        new_data.iloc[i,3*point+1] = cordinate[1]
        new_data.iloc[i,3*point+2] = cordinate[2]
        p.resetBasePositionAndOrientation(point,cordinate,[0,0,0,1])
    # p.addUserDebugPoints(point_cordinate,color_list,pointSize = 4., lifeTime = .0001)
    t.sleep(.05)
    p.stepSimulation()
print(new_data.head())
print(new_data.keys())



def cal_angle(vec1:np.ndarray,vec2:np.ndarray):
    norm1, norm2 = np.linalg.norm(vec1,axis=1), np.linalg.norm(vec2,axis=1)
    return np.arccos(np.sum(vec1*vec2,axis=1)/(norm1*norm2))


# new_data.to_csv('bipedal_env\\trajectories\\motion_cordinate.csv',index=False)
l_thigh = - data.iloc[:,3:6].to_numpy()     + data.iloc[:,6:9].to_numpy()
r_thigh = - data.iloc[:,15:18].to_numpy()   + data.iloc[:,18:21].to_numpy()
l_bicep = - data.iloc[:,6:9].to_numpy()     + data.iloc[:,9:12].to_numpy()
r_bicep = - data.iloc[:,18:21].to_numpy()   + data.iloc[:,21:24].to_numpy()
l_feet  = - data.iloc[:,9:12].to_numpy()    + data.iloc[:,12:15].to_numpy()
r_feet  = - data.iloc[:,21:24].to_numpy()   + data.iloc[:,24:27].to_numpy()
lleg_angle      = cal_angle(l_thigh*np.array([[1,0,1]]),l_thigh)
rleg_angle      = cal_angle(r_thigh*np.array([[1,0,1]]),r_thigh)
lthigh_angle    = cal_angle(l_thigh*np.array([[0,1,1]]),l_thigh)
rthigh_angle    = cal_angle(r_thigh*np.array([[0,1,1]]),r_thigh)
import matplotlib.pyplot as plt
plt.plot(lleg_angle)
plt.plot(rleg_angle)
plt.plot(lthigh_angle)
plt.plot(rthigh_angle)
plt.show()
# print(l_thigh)