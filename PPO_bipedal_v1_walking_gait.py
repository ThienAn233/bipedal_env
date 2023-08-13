import torch
import torch.nn as nn
import numpy as np
import time as t
import bipedal_walker_env_v1 as bpd
from torch.utils.data import Dataset, DataLoader
from torchrl.modules import TanhNormal
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
class PPO_bipedal_walker_train():
    def __init__(self,
                # Global variables
                PATH = None,
                load_model = None,
                envi = bpd,
                log_data = True,
                save_model = True,
                render_mode = False,
                thresh = 1.5,

                alpha = 0.5,
                epsilon = 0.2,
                explore = 1e-4,
                gamma = .99,
                learning_rate = 1e-4,
                number_of_robot = 9,
                epochs = 500,
                data_size = 1000,
                batch_size = 5000,
                reward_index = np.array([[0, 0, 0, 1, 1, 1]]),
                seed = 1107,
                mlp = None,

                # local variables
                # Seed & devices
                action_space = 8,
                observation_space = 29,
                device = None):

                    
                    
        # Global variables
        self.PATH = PATH
        self.load_model = load_model
        self.envi = envi
        self.log_data = log_data
        self.save_model = save_model
        self.render_mode = render_mode
        self.thresh = thresh
        self.alpha = alpha
        self.epsilon = epsilon
        self.explore = explore
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.number_of_robot = number_of_robot
        self.epochs = epochs
        self.data_size = data_size
        self.batch_size = batch_size
        self.reward_index = reward_index
        self.seed = seed
        self.mlp = mlp


                    
        # local variables
        # Seed & devices
        self.action_space = action_space
        self.observation_space = observation_space + action_space
        self.device = device
        
        
        
        # Load model and device
        if load_model:
            self.model_path = PATH + '//models//PPO//' + load_model
            self.optim_path = PATH + '//models//PPO//' + load_model + 'optim'
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        print(f'Using seed: {self.seed}')
        if self.device:
            pass
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Using device: ', self.device)
        # Tensor board
        if self.log_data:
            self.writer = SummaryWriter(PATH + '//runs//PPO//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime()))
            # Envs setup
        
        self.env = envi.bipedal_walker(num_robot=self.number_of_robot,render_mode=self.render_mode, show_traj=True,floor=False)
        self.env.terrainHeight = [0, .02]
        print('env is ready!')
        print(f'action space of {number_of_robot} robot is: {action_space}')
        print(f'observation sapce of {number_of_robot} robot is: {observation_space}')
        
        
        
        # Setup MLP
        if self.mlp:
            self.mlp.to(self.device)
            pass
        else:
            class MLP(nn.Module):
                def __init__(self):
                    super(MLP,self).__init__()
                # nn setup
                    lin1 = nn.Linear(observation_space,500)
                    torch.nn.init.xavier_normal_(lin1.weight,gain=1)
                    lin2 = nn.Linear(500,100)
                    torch.nn.init.xavier_normal_(lin2.weight,gain=1)
                    lin3 = nn.Linear(100,action_space)
                    torch.nn.init.constant_(lin3.weight,0.)
                    lin4 = nn.Linear(100,action_space)
                    torch.nn.init.constant_(lin4.weight,0.)
                    lin5 = nn.Linear(100,1)
                    torch.nn.init.xavier_normal_(lin5.weight,gain=1)
                    self.mean = nn.Sequential(
                        lin1,
                        nn.LeakyReLU(.2),
                        lin2,
                        nn.LeakyReLU(.2),
                        lin3,
                    )
                    self.var = nn.Sequential(
                        lin1,
                        nn.LeakyReLU(.2),
                        lin2,
                        nn.LeakyReLU(.2),
                        lin4,
                    )
                    self.critic = nn.Sequential(
                        lin1,
                        nn.LeakyReLU(.2),
                        lin2,
                        nn.LeakyReLU(.2),
                        lin5,
                    )
                def forward(self,input):
                    return self.mean(input),self.var(input),self.critic(input)
            self.mlp = MLP().to(self.device)
        print('MLP is ready!')
        print('params: ',sum(i.numel() for i in self.mlp.parameters()) )
        ###
        ### Normalize the return and obs
        self.mlp.eval()
        with torch.no_grad():
                data = self.get_data_from_env()
        data = custom_dataset(data,self.data_size,self.number_of_robot,self.gamma)
        self.qua_var_mean = torch.var_mean(data.local_return,dim=0)
        self.val_var_mean = torch.var_mean(data.local_values,dim=0)
        sns.kdeplot(data=data.local_action.squeeze())
        plt.show()
        print(f'quality var mean: {self.qua_var_mean}')
        print(f'values var mean: {self.val_var_mean}')
        ###


        # optim setup
        self.mlp_optimizer = torch.optim.Adam(self.mlp.parameters(),lr = self.learning_rate)
        if load_model:
            self.mlp.load_state_dict(torch.load(self.model_path,map_location=self.device))
            self.mlp_optimizer.load_state_dict(torch.load(self.optim_path,map_location=device))
        else:
            pass
        self.mlp_optimizer.param_groups[0]['lr'] = self.learning_rate
        print(self.mlp_optimizer.param_groups[0]['lr'])

       
    
    #helper functions
    
    def get_actor_critic_action_and_values(self,obs,eval=True):
        logits, var, values = self.mlp(obs)
        old_shape = logits.shape
        logits, var = logits.view(*logits.shape,1), var.view(*var.shape,1)
        probs = TanhNormal(loc = logits, scale=.2*nn.Sigmoid()(var),max=np.pi/4,min=-np.pi/4)
        # probs = Normal(loc = logits,scale=0.5*nn.Sigmoid()(var))
        if eval is True:
            action = probs.sample()
            return action.view(old_shape), probs.log_prob(action), values
        else:
            action = eval
            dummy_action = probs.sample((100,))
            return action.view(old_shape), probs.log_prob(action.view(logits.shape)), -probs.log_prob(dummy_action).mean(dim=0), values

    def get_data_from_env(self):
        ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
        local_observation = []
        local_action = []
        local_logprob = []
        local_reward = []
        local_timestep = []
        local_values = []
        
        observation = self.env.get_obs()[0]
        # previous_action = self.env.previous_pos
        # observation = np.hstack([observation,previous_action])
        
        timestep = np.array(self.env.time_steps_in_current_episode)
        local_timestep.append(torch.Tensor(timestep.copy()))
        for i in range(self.data_size) :
            # act and get observation 
            action, logprob, values = self.get_actor_critic_action_and_values(torch.Tensor(observation).to(self.device))
            action, logprob = action.cpu(), logprob.cpu()
            local_observation.append(torch.Tensor(observation))
            local_action.append(torch.Tensor(action))
            local_logprob.append(torch.Tensor(logprob))
            self.env.sim(np.array(action),real_time=False)
            observation, reward, info= self.env.get_obs(train=True)
            # stacking obs and previous action
            # print(reward[0])
            # previous_action = action
            # observation = np.hstack([observation,previous_action])
            reward = np.sum(reward*self.reward_index,axis=-1)
            truncated = info
            
            # save var
            local_reward.append(torch.Tensor(reward))
            local_values.append(torch.Tensor(values))
            timestep = np.array(self.env.time_steps_in_current_episode)
            local_timestep.append(torch.Tensor(timestep.copy()))
        return local_observation, local_action, local_logprob, local_reward, local_timestep, local_values

    def train(self):
        best_reward = 0
        for epoch in range(self.epochs):
            mlp = self.mlp.eval()
            # Sample data from the environment
            with torch.no_grad():
                data = self.get_data_from_env()
            dataset = custom_dataset(data,self.data_size,self.number_of_robot,self.gamma)
            dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
            for iteration, data in enumerate(dataloader):
                mlp = mlp.train()
                
                obs, action, logprob, quality, reward = data
                # Normalize return
                quality = (quality-self.qua_var_mean[1])/self.qua_var_mean[0]**.5
                obs, action, logprob, quality, reward = obs.to(self.device), action.to(self.device), logprob.to(self.device), quality.to(self.device), reward.to(self.device)
                next_action, next_logprob, entropy, value = self.get_actor_critic_action_and_values(obs,eval=action)
                # Normalize values
                value = (value-self.val_var_mean[1])/self.val_var_mean[0]**.5
                # Train models
                self.mlp_optimizer.zero_grad()
                prob_ratio = torch.exp(next_logprob-logprob)
                advantage = quality-value
                critic_loss = (advantage**2).mean()
                entropy_loss = entropy.mean()
                actor_loss = - torch.min( prob_ratio*advantage , torch.clamp(prob_ratio, 1-self.epsilon, 1+self.epsilon)*advantage ).mean() - self.explore*entropy_loss
                loss = critic_loss + self.alpha*actor_loss
                loss.backward()
                self.mlp_optimizer.step()
                
                #save model
                if self.save_model:
                    if (quality.mean().item()>best_reward and quality.mean().item() > self.thresh) | ((epoch*(len(dataloader))+iteration) % 250 == 0):
                        best_reward = quality.mean().item()
                        torch.save(mlp.state_dict(), self.PATH+'models\\PPO\\'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(quality.mean().item(),2)))
                        torch.save(self.mlp_optimizer.state_dict(), self.PATH+'models\\PPO\\'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(quality.mean().item(),2))+'optim')
                        print('saved at: '+str(round(quality.mean().item(),2)))
                
                # logging info
                if self.log_data:
                    self.writer.add_scalar('Eval/minibatchreward',reward.mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Eval/minibatchreturn',quality.mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Train/entropyloss',entropy_loss.item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Train/criticloss',critic_loss.detach().mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Train/actorloss',actor_loss.detach().mean().item(),epoch*(len(dataloader))+iteration)
                print(f'[{epoch}]:[{self.epochs}]|| iter [{epoch*(len(dataloader))+iteration}]: rew: {round(reward.mean().item(),2)} ret: {round(quality.mean().item(),2)} cri: {critic_loss.detach().mean().item()} act: {actor_loss.detach().mean().item()} entr: {entropy_loss.detach().item()}')
        torch.save(mlp.state_dict(), self.PATH+'models\\PPO\\'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime()))
        torch.save(self.mlp_optimizer.state_dict(), self.PATH+'models\\PPO\\'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'optim')




class custom_dataset(Dataset):
    
    def __init__(self,data,data_size,number_of_robot,gamma):
        self.data_size = data_size
        self.number_of_robot = number_of_robot
        self.gamma = gamma
        self.obs, self.action, self.logprob, self.reward, self.timestep, self.values = data        
        self.local_return = [0 for i in range(data_size)]
        self.local_return = torch.vstack(self.get_G()).view(-1,1)
        self.local_observation = torch.vstack(self.obs)
        self.local_action = torch.vstack(self.action)
        self.local_logprob = torch.vstack(self.logprob)
        self.local_reward = torch.vstack(self.reward).view(-1,1)
        self.local_values = torch.vstack(self.values).view(-1,1)
        # print(self.local_observation)
        # print(self.local_action)
        # print(self.local_logprob)
        # print(self.local_return)
        # print(self.local_reward)

    def __len__(self):
        return self.data_size*self.number_of_robot
    
    def __getitem__(self, index):
        return self.local_observation[index], self.local_action[index], self.local_logprob[index], self.local_return[index], self.local_reward[index]
    
    def isnt_end(self, i):
        ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
        return self.timestep[i] != 0
    
    def get_G(self):
        ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
        for  i in range(self.data_size-1,-1,-1):
            if i == self.data_size-1:
                self.local_return[i] = self.reward[i]
            else:
                self.local_return[i] = self.reward[i] + self.isnt_end(i)*self.gamma*self.local_return[i+1]
        return self.local_return   
    
trainer = PPO_bipedal_walker_train(
                                PATH='bipedal_env//',
                                load_model='2023-08-14-02-21-14',
                                number_of_robot = 9,
                                learning_rate = 1e-4,
                                data_size = 400,
                                batch_size = 1800,
                                epochs=2000,
                                thresh=1,
                                explore = 1e-6,
                                alpha = 0.5,
                                epsilon = 0.1,
                                log_data = True,
                                save_model = True,
                                render_mode= False)
trainer.train()