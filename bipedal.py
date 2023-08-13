import pybullet as p
import pybullet_data
import numpy as np
import time as t
import matplotlib.pyplot as plt



# Configure-able variables
PATH = 'C:\\Users\\Duc Thien An Nguyen\\Desktop\\my_collections\\Python\\bipedal_env\\bipedal.urdf'
sleep_time = 1./2400.
target_radius = [0,5]
target_height = [0.2,0.5]
initialHeight = 1.#0.3752
initialOri = [0,0,1,0]
jointId_list = []
jointName_list = []
jointRange_list = []
jointMaxForce_list = []
jointMaxVeloc_list = []
debugId_list = []
ori_debug_list = []
temp_debug_value = []
mode = p.POSITION_CONTROL
physicsClient = p.connect(p.GUI)
# physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Constants
g = (0,0,-9.81) 
pi = np.pi

pos_list = []
set_list = []


# Settup the environment and print out some variables
print('-----------------------------------')

# Sample and calculate a random point as target and inital position
random_radius = np.random.uniform(*target_radius,size=2)
random_heights = np.random.uniform(*target_height)
random_angle = np.random.uniform([0,2*pi],size=2)
random_cordinates = np.vstack([np.sin(random_angle)*random_radius, np.cos(random_angle)*random_radius, np.array([initialHeight,random_heights])])
initialPos = random_cordinates[:,0]
random_target = random_cordinates[:,1]
p.addUserDebugPoints([random_target],[(1,0,0)],pointSize = 10)


p.setGravity(*g)
robotId = p.loadURDF(PATH,[0.,0.,initialHeight],initialOri)
planeId = p.loadURDF('plane.urdf')
number_of_joints = p.getNumJoints(robotId)

print(f'Robot id: {robotId}')
print(f'number of robot joints: {number_of_joints}')

for jointIndex in range(0,number_of_joints):
    data = p.getJointInfo(robotId, jointIndex)
    jointId_list.append(data[0])                                                                                # Create list to store joint's Id
    jointName_list.append(str(data[1]))                                                                         # Create list to store joint's Name
    jointRange_list.append((data[8],data[9]))                                                                   # Create list to store joint's Range
    jointMaxForce_list.append(data[10])                                                                         # Create list to store joint's max Force
    jointMaxVeloc_list.append(data[11])                                                                         # Create list to store joint's max Velocity
    debugId_list.append(p.addUserDebugParameter(str(data[1]), rangeMin = data[8], rangeMax = data[9], ))        # Add debug parameters to manually control joints
    p.enableJointForceTorqueSensor(robotId,jointIndex,True)
    print(f'Id: {data[0]}, Name: {str(data[1])}, Range: {(data[8],data[9])}, DebugId: {debugId_list[-1]}')
p.setJointMotorControlArray(robotId,jointId_list,mode)
print(f'Control mode is set to: {"Velocity" if mode==0 else "Position"}')
print('-----------------------------------')


p.resetBasePositionAndOrientation(robotId,[0.,0.,initialHeight],initialOri)
p.resetBaseVelocity(robotId,[0,0,0],[0,0,0])
for jointId in jointId_list:
            p.resetJointState(bodyUniqueId=robotId,jointIndex=jointId,targetValue=0,targetVelocity=0)
for i in range(3):
    ori_debug_list.append(p.addUserDebugParameter('quat'+str(i),rangeMin = -9, rangeMax = 9))
previous_pos = np.zeros((len(jointId_list)))

# Start the simulation loop
for i in range(10000):
    
    ## ACTION OF THE AGENT
    # Reset readed variables and feed them to motors
    temp_debug_value = []
    for Id in debugId_list:
        temp_debug_value.append(p.readUserDebugParameter(Id))
    set_list.append(temp_debug_value[0]*100)
    filtered_action = previous_pos*.8 + np.array(temp_debug_value)*.2
    p.setJointMotorControlArray(robotId,
                                jointId_list,
                                mode,
                                targetPositions = filtered_action,
                                forces = jointMaxForce_list, 
                                targetVelocities = jointMaxVeloc_list,
                                positionGains = np.ones_like(temp_debug_value)*.5,
                                # velocityGains = np.ones_like(temp_debug_value)*0.,        
                                )
    previous_pos = np.array(temp_debug_value)
    temp_debug_ori_list = []
    for Id in ori_debug_list:
        temp_debug_ori_list.append(p.readUserDebugParameter(Id))
    w, x, y = temp_debug_ori_list
    ori_list = np.array([x,y,1,w])/(x**2+y**2+1+w**2)**.5
    ori_list_visual = ori_list[:3]/np.linalg.norm(ori_list[:3])
    # print(ori_list)
   
    p.addUserDebugLine(np.array([0.,0.,initialHeight])-ori_list_visual,np.array([0.,0.,initialHeight])+ori_list_visual,lineWidth = 10, lifeTime =1, lineColorRGB = [1,0,0])
    # ori_list=[0,0,-1,0]
    p.resetBasePositionAndOrientation(robotId,[0.,0.,initialHeight],ori_list)
    linkinfo = p.getLinkState(robotId,2)
    # print(linkinfo[0])
    # Step the simulation
    p.stepSimulation()
    ## OBSERVATION OF THE AGENT
    # Joints obs
    temp_obs_value = []
    for Id in jointId_list:
        for _ in p.getJointState(robotId,Id)[:2]:
            temp_obs_value.append(_)
    link_info = p.getLinkStates(robotId,jointId_list)
    joint_info = p.getJointState(robotId,0)[2][:3]
    pos_list.append(joint_info)
    for i in p.getBaseVelocity(robotId):
        for _ in i:
            temp_obs_value.append(_)
    base_inf =  p.getBasePositionAndOrientation(robotId)
    temp_obs_value.append(base_inf[0][-1])
    for _ in p.getEulerFromQuaternion(base_inf[-1]):
        temp_obs_value.append(_)
    x, y= p.getEulerFromQuaternion(base_inf[1])[:2]
    ori = np.tan(x/2)**2+np.tan(y/2)**2
    linear_velo, angular_velo = p.getBaseVelocity(robotId)
    # for Id in (3,7):
    #         feet_pos, feet_ori = p.getLinkState(robotId,Id)[:2]
    #         roll, pitch, yaw = p.getEulerFromQuaternion(feet_ori)
    #         x = np.cos(yaw)*np.cos(pitch)
    #         y = np.sin(yaw)*np.cos(pitch)
    #         z = np.sin(pitch)

    #         dir = np.array([x,y,z])
    #         p.addUserDebugLine(feet_pos, feet_pos - dir , lifeTime =1.)
    # print(base_inf[0][-1])
    t.sleep(sleep_time)
plt.plot(set_list)
plt.plot(pos_list)
plt.show()