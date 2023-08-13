import pybullet as p 
import pybullet_data
import gymnasium as gym
import numpy as np
import time as t
import os

# Global variables (Unchange-able)
g = (0,0,-9.81)                                                                                     # Gravity
pi = np.pi                                                                                          # Pi
# Local variables (Change-able)
simulation_step = 500                                                                               # Number of simulation step
sleep_time = 1./240.                                                                                # Sleep time btw step
total_time = simulation_step*sleep_time                                                             # Total time
startPos = np.random.normal([0,0,1],scale=[0.5,0.5,0.5])                                            # Position of obj
startOrientation = p.getQuaternionFromEuler(np.random.normal([0,0,0],scale=[pi/4,pi/4,pi/4]))       # Ori of obj
print(f'Total time:{round(total_time,2)}s++')


# Connect to the physics client
physicsClient = p.connect(p.GUI)
# Set the search Path
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Set gravity
p.setGravity(*g)
# Cearte the global plane
planeId = p.loadURDF('plane.urdf')
#  Defy the object properties
boxId = p.loadURDF("r2d2.urdf",startPos, startOrientation)
p.resetBasePositionAndOrientation(boxId,startPos,startOrientation)
number_of_joints = p.getNumJoints(boxId)
print(f'Number of joints is {number_of_joints}')


# Start the simulation
print('--------------------------------------------------------------')
for i in range(simulation_step):
    p.stepSimulation()
    t.sleep(sleep_time)
print('--------------------------------------------------------------')


# Get info of object
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
# Disconect physics client
p.disconnect()
print('Done')