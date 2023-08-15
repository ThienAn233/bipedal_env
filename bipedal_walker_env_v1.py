import pybullet as p 
import pybullet_data
import numpy as np
import pandas as pd
import time as t 



class bipedal_walker():
    
    def __init__(self,
                 max_length = 400,
                 num_step = 10,
                 render_mode = None,
                 robot_file = None,
                 show_traj = None,
                 num_robot = 9,
                 seed = 0,
                 floor = True):
        
        # Configure-able variables
        self.num_step = num_step
        self.max_length = max_length
        self.render_mode = render_mode
        self.robot_file = robot_file
        if render_mode:
            self.physicsClient = p.connect(p.GUI)
            self.sleep_time = 1./240.
        else:
            self.physicsClient = p.connect(p.DIRECT)
            self.sleep_time = 1./240.
        # p.setTimeStep(self.sleep_time*self.num_step,self.physicsClient)
        if robot_file:
            self.robot_file = robot_file
        else:
            self.robot_file = 'bipedal_env//bipedal.urdf'
        self.floor = floor
        self.show_traj = show_traj
        self.num_robot = num_robot
        self.target_height = [0.4,0.9]
        self.initialVel = [0, .1]
        self.initialMass = [0, .2]
        self.initialPos = [0, .1]
        self.initialFriction = [0, .3]
        self.terrainHeight = [0, .05]
        self.terrainScale = [.05, .05, 1]
        self.initialHeight = .7 + self.terrainHeight[-1]
        self.robotId_list = []
        self.jointId_list = []
        self.jointName_list = []
        self.jointRange_list = []
        self.jointMaxForce_list = []
        self.jointMaxVeloc_list = []
        self.trajectories = None
        self.current_traj_step = [0 for i in range(self.num_robot)]
        self.mode = p.POSITION_CONTROL
        self.seed = seed
        np.random.seed(self.seed)
        
        # Constants (DO NOT TOUCH)
        self.g = (0,0,-9.81) 
        self.pi = np.pi
        self.time_steps_in_current_episode = [1 for _ in range(self.num_robot)]
        self.vertical = np.array([0,0,1])
        self.terrain_shape = [10, self.num_robot]
        self.num_points = 5
        self.x_sc = 25
        self.y_sc = 18
        self.z_sc = 25
               
        # Settup the environment and print out some variables
        print('-----------------------------------')
        
        # print(f'ENVIRONMENT STARTED WITH SEED {self.seed}')
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId = self.physicsClient)
        # load trajectories files
        trajectory_path = 'bipedal_env\\trajectories\\motion_01.csv'
        self.trajectories = pd.read_csv(trajectory_path)
            
        # Load URDF and print info
        p.setGravity(*self.g, physicsClientId = self.physicsClient)
        self.get_init_pos()
        for pos in self.corr_list:
            self.robotId_list.append(p.loadURDF(self.robot_file, physicsClientId = self.physicsClient,basePosition=pos,baseOrientation=[0,0,1,0],useFixedBase=0 if self.floor else -1))
        if self.floor:
            self.sample_terrain()
        # self.planeId = p.loadURDF('plane.urdf', physicsClientId = self.physicsClient)
        self.number_of_joints = p.getNumJoints(self.robotId_list[0], physicsClientId = self.physicsClient)

        for jointIndex in range(0,self.number_of_joints):
            data = p.getJointInfo(self.robotId_list[0], jointIndex, physicsClientId = self.physicsClient)
            self.jointId_list.append(data[0])                                                                                # Create list to store joint's Id
            self.jointName_list.append(str(data[1]))                                                                         # Create list to store joint's Name
            self.jointRange_list.append((data[8],data[9]))                                                                   # Create list to store joint's Range
            self.jointMaxForce_list.append(data[10])                                                                         # Create list to store joint's max Force
            self.jointMaxVeloc_list.append(data[11])                                                                         # Create list to store joint's max Velocity
            # print(f'Id: {data[0]}, Name: {str(data[1])}, Range: {(data[8],data[9])}')
        self.robotBaseMassandFriction = [p.getDynamicsInfo(self.robotId_list[0],-1,physicsClientId = self.physicsClient)[0], p.getDynamicsInfo(self.robotId_list[0],self.jointId_list[-1],physicsClientId = self.physicsClient)[1]]
        print(f'Robot mass: {self.robotBaseMassandFriction[0]} and friction on feet: {self.robotBaseMassandFriction[1]}')
        for robotId in self.robotId_list:
            p.setJointMotorControlArray(robotId,self.jointId_list,self.mode, physicsClientId = self.physicsClient)
            for joints in self.jointId_list:
                p.enableJointForceTorqueSensor(robotId,joints,True,physicsClientId = self.physicsClient)
            self.sample_target(robotId)
        print(f'Robot position loaded, force/torque sensors enable')
        self.previous_pos = np.zeros((self.num_robot,len(self.jointId_list)))
        self.reaction_force = np.zeros((self.num_robot,len(self.jointId_list),3))
        
        # viz_traj
        if self.show_traj:
            self.point_list = [p.loadURDF('bipedal_env\\target.urdf') for _ in range(self.num_points*self.num_robot)]
        else:
            self.point_list = []
        
        print('-----------------------------------')
        
    def get_init_pos(self):
        nrow = int(self.num_robot)
        x = np.linspace(-(nrow-1)/2,(nrow-1)/2,nrow)
        xv,yv = np.meshgrid(0,x)
        xv, yv = np.hstack(xv), np.hstack(yv)
        zv = self.initialHeight*np.ones_like(xv)
        self.corr_list = np.vstack((xv,yv,zv)).transpose()
        
    def sim(self,action,real_time = False):
        filtered_action = self.previous_pos*.8 + action*.2
        self.previous_pos = action
        
        self.time_steps_in_current_episode = [self.time_steps_in_current_episode[i]+1 for i in range(self.num_robot)]
        self.step_trajectories()
        for _ in range(self.num_step):
            self.act(filtered_action)
            p.stepSimulation( physicsClientId = self.physicsClient)
            if real_time:
                t.sleep(self.sleep_time*self.num_step)
            
    def get_obs(self,train=True):
        
        temp_obs_value = []
        temp_info = []
        temp_reward_value = []

        for robotId in self.robotId_list:
            # GET OBSERVATION
            temp_obs_value += [self.get_all_obs(robotId)]

            # GET INFO
            # Check weather the target is reached, if no, pass, else sammple new target
            if train:
                temp_info += [self.auto_reset(robotId,temp_obs_value[-1])]
            # GET REWARD
            temp_reward_value += [self.get_reward_value(temp_obs_value[-1],robotId)]
        

        return np.array(temp_obs_value), np.array(temp_reward_value), np.array(temp_info)
    
    
    def act(self,action):
        for robotId in self.robotId_list:
            p.setJointMotorControlArray(robotId,self.jointId_list,
                                        self.mode,
                                        targetPositions = action[robotId], 
                                        forces = self.jointMaxForce_list, 
                                        targetVelocities = self.jointMaxVeloc_list, 
                                        positionGains = np.ones_like(self.jointMaxForce_list)*.2,
                                        # velocityGains = np.ones_like(self.jointMaxForce_list)*1,
                                        physicsClientId = self.physicsClient)
    
    def close(self):
        p.disconnect(physicsClientId = self.physicsClient)
    
    def sample_target(self,robotId):
        random_Ori = [0,0,1,0]
        # Sample new position
        pos = self.corr_list[robotId] + np.array(list(np.random.uniform(*self.initialPos,size=2))+[0])
        if self.floor:
            p.resetBasePositionAndOrientation(robotId, pos, random_Ori, physicsClientId = self.physicsClient)
            # Sample new velocity
            init_vel = np.random.normal(loc = self.initialVel[0],scale = self.initialVel[1],size=(3))
            p.resetBaseVelocity(robotId,init_vel,[0,0,0],physicsClientId=self.physicsClient)
            # Sample new base mass
            new_mass = self.robotBaseMassandFriction[0] + np.random.uniform(*self.initialMass)
            p.changeDynamics(robotId,-1,new_mass)
            # Sample new feet friction
            new_friction = self.robotBaseMassandFriction[1] + np.random.uniform(*self.initialFriction)
            for i in (3,7):
                p.changeDynamics(robotId,i,lateralFriction=new_friction)
            for jointId in self.jointId_list:
                p.resetJointState(bodyUniqueId=robotId,jointIndex=jointId,targetValue=0,targetVelocity=0,physicsClientId=self.physicsClient)
    
    def sample_terrain(self):
        numHeightfieldRows = int(self.terrain_shape[0]/self.terrainScale[0])
        numHeightfieldColumns = int(self.terrain_shape[1]/self.terrainScale[1])
        heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
        for j in range (int(numHeightfieldColumns/2)):
            for i in range (int(numHeightfieldRows/2) ):
                height = round(np.random.uniform(*self.terrainHeight),2)
                heightfieldData[2*i+2*j*numHeightfieldRows]=height
                heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
                heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
                heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
        terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=self.terrainScale, heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns, physicsClientId=self.physicsClient)
        self.terrainId = p.createMultiBody(0, terrainShape, physicsClientId=self.physicsClient,useMaximalCoordinates =True)
        p.resetBasePositionAndOrientation(self.terrainId,[-4.5,0,0], [0,0,0,1], physicsClientId=self.physicsClient)
        self.textureId = p.loadTexture("heightmaps/gimp_overlay_out.png")
        p.changeVisualShape(self.terrainId, -1, textureUniqueId = self.textureId)
    
    def get_distance_and_ori_and_velocity_from_target(self,robotId):
        temp_obs_value = []
        
        # Get cordinate in robot reference 
        base_position, base_orientation =  p.getBasePositionAndOrientation(robotId, physicsClientId = self.physicsClient)[:2]
        base_position = [-base_position[1]+self.corr_list[robotId][1]] + [base_position[-1]]
        temp_obs_value += [ *base_position]
        temp_obs_value += [*base_orientation]
        # Get base linear and angular velocity
        linear_velo, angular_velo = p.getBaseVelocity(robotId, physicsClientId = self.physicsClient)
        temp_obs_value += [*linear_velo, *angular_velo]
        
        return temp_obs_value
    
    def get_joints_values(self,robotId):
        temp_obs_value = []
        # Get joints reaction force for reward
        # Get joints position and velocity
        for Id in self.jointId_list:
            temp_obs_value += [*p.getJointState(robotId,Id, physicsClientId = self.physicsClient)[:2]]
            self.reaction_force[robotId,Id,:] = p.getJointState(robotId,Id,physicsClientId = self.physicsClient)[2][:3]
        return temp_obs_value
    
    def get_trajectory_values(self,robotId):
        temp_obs_value = []
        progress = self.current_traj_step[robotId]/(self.trajectories.shape[0]//5)
        temp_obs_value += [progress]
        return temp_obs_value
    
    def get_all_obs(self,robotId):
        temp_obs_value = []
        
        # Base position state
        base_info = self.get_distance_and_ori_and_velocity_from_target(robotId)
        
        # Joints state
        joints_info = self.get_joints_values(robotId)

        # Trajec state
        trajec_info = self.get_trajectory_values(robotId)
        
        # Full observation
        temp_obs_value += [
                        *base_info,
                        *joints_info,
                        *trajec_info,
                        ]

        return temp_obs_value
    
    def truncation_check(self,height,dir,robotId):
        return  (self.time_steps_in_current_episode[robotId]>self.max_length) | (self.target_height[0] > height) | (np.abs(dir)>0.25)
    
    def auto_reset(self,robotId,obs):
        height, dir = obs[1], obs[0]
        truncation = self.truncation_check(height,dir,robotId)
        if truncation:
            self.sample_target(robotId)
            self.current_traj_step[robotId] = 0
            self.time_steps_in_current_episode[robotId] = 0
            self.previous_pos[robotId] = np.zeros((len(self.jointId_list)))
        return truncation
    
    def step_trajectories(self):
        for i in self.robotId_list:
            self.current_traj_step[i] +=1
            if self.current_traj_step[i] +1 > self.trajectories.shape[0]//5:
                self.current_traj_step[i] = 0
        return
    
    def get_trajectories_corr(self,robotId):
        point_cordinate_list = []
        x, y ,z = self.x_sc, self.y_sc, self.z_sc
        for point in range(self.num_points):
            dt = self.current_traj_step[robotId]*5
            data = self.trajectories
            robot_corr_list = np.array([*p.getBasePositionAndOrientation(robotId)[0][:2]]+[self.initialHeight])
            base_cordinate = np.array((-data.iloc[dt,2]/x,data.iloc[dt,0]/y,data.iloc[dt,1]/z))
            point_cordinate = np.array((-data.iloc[dt,3*point+2]/x,data.iloc[dt,3*point]/y,data.iloc[dt,3*point+1]/z))
            point_cordinate -= base_cordinate - robot_corr_list
            if self.show_traj:
                p.resetBasePositionAndOrientation(self.point_list[robotId*self.num_points+point],point_cordinate,[0,0,0,1])
            point_cordinate_list+= [point_cordinate]
        return point_cordinate_list
    
    def calculate_imi_reward(self,robotId):
        joint_list_idx = [2,3,6,7]
        robot_corr_list = [p.getBasePositionAndOrientation(robotId)[0]]
        traj_corr_list = np.array(self.get_trajectories_corr(robotId))
        for joint in joint_list_idx:
            robot_corr_list += [p.getLinkState(robotId,joint)[0]]
        robot_corr_list = np.array(robot_corr_list)
        # print(np.linalg.norm(robot_corr_list-traj_corr_list,axis=1))
        reward = -4*np.linalg.norm(robot_corr_list-traj_corr_list,axis=1).sum()
        return reward
    
    def get_reward_value(self,obs,robotId):
        # Reward for high speed in x direction
        speed = -10*obs[6]

        # Reward for being in good y direction
        align = -50*obs[0]**2
        # -10*obs[0]**2
        
        # Reward for being high
        high = -100*(-obs[1]+.65) if obs[1]<.65 else 0
        
        # Reward for surviving 
        surv = 30
        
        # Reward for minimal force
        force = (-1e-5)*((self.reaction_force[robotId,:]**2).sum())
        
        # Reward for immitation
        immit = 5*self.calculate_imi_reward(robotId)
        return [speed, align, high, surv, force, immit ]
        
# # TEST ###
# env = bipedal_walker(render_mode='human',num_robot=5,show_traj=False,floor=True)
# for _ in range(1200):
#     env.sim(np.random.uniform(-.1,.1,(env.num_robot,8)),real_time=True)
#     obs,rew,inf = env.get_obs()
#     # print(rew[0])
#     # print(obs[0])
#     # print(env.previous_pos)
#     # print(env.delta_pos)
# env.close()