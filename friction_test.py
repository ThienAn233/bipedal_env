import pybullet as p
import pybullet_data
import time
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
sphere = p.loadURDF("sphere2.urdf",[0,0,1])
info = p.getDynamicsInfo(sphere,-1)
print(info)
p.resetBaseVelocity(sphere,linearVelocity=[1,0,0])
# p.changeDynamics(sphere,-1,lateralFriction=0.5)
p.changeDynamics(sphere,-1,rollingFriction=1)
p.setGravity(0,0,-10)
for i in range (1000):
    p.stepSimulation()
    time.sleep(1./240.)
    vel = p.getBaseVelocity(sphere)
    # assert vel[0][0]>1e-10, 'error'
    # assert vel[0][1]>1e-10, 'error'
    # assert vel[0][2]>1e-10, 'error'
    # assert vel[1][0]>1e-10, 'error'
    # assert vel[1][1]>1e-10, 'error'
    # assert vel[1][2]>1e-10, 'error'
p.disconnect() 