import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym 
import time as t
import bipedal_walker_env_v1 as bpd
from torch.distributions.normal import Normal
from torchrl.modules import TanhNormal
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Global variables
model_path = 'bipedal_env\\models\\PPO\\2023-07-05-04-45-22_best_0.89'
# model_path = 'C:\\Users\\Duc Thien An Nguyen\\Desktop\\my_collections\\Python\\bipedal_env\\models\\PPO2023-06-21-01-12-22_best_0.89'
log_data = False
save_model = False
thresh = 5

epsilon = 0.2
explore = 2e-2
gamma = 0.8
learning_rate = 0.1
number_of_envs = 1
epochs = 1
data_size = 1000
batch_size = 1000
rewrad_index = np.array([[0.25, 0.25, 0.25, 0.25]])
seed = 0

# local variables
    # Seed & devices
action_space = 6
observation_space = 25
torch.manual_seed(seed)
np.random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device: ', device)
    # Tensor board
if log_data:
    writer = SummaryWriter('C:\\Users\\Duc Thien An Nguyen\\Desktop\\my_collections\\Python\\bipedal_env\\runs\\PPO\\'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime()))
    # Envs setup
env = bpd.bipedal_walker(render_mode='human',num_robot=1)
print(f'action space of {number_of_envs} envs is: {action_space}')
print(f'observation sapce of {number_of_envs} envs is: {observation_space}')
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
    # nn setup
        self.actor = nn.Sequential(
            nn.Linear(observation_space+action_space,500),
            nn.Tanh(),
            nn.Linear(500,action_space*2),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_space+action_space,500),
            nn.Tanh(),
            nn.Linear(500,1)
        )
    def forward(self,input):
        return self.actor(input),self.critic(input)


#helper functions
def get_actor_critic_action_and_values(obs,eval=True):
    logits, values = mlp(obs)
    logits = logits.view(*logits.shape,1)
    # print(logits.shape)
    probs = TanhNormal(loc = logits[:action_space], scale=.1*nn.Sigmoid()(logits[action_space:]),max=np.pi/2,min=-np.pi/2)
    # probs = TanhNormal(loc = (torch.pi/2)*nn.Tanh()(logits[:,:self.action_space]),scale=0.5*nn.Sigmoid()(logits[:,self.action_space:]))
    if eval is True:
        action = probs.sample()
        # print(probs.log_prob(action).shape)
        return action, -probs.log_prob(action)
    else:
        action = eval
        # print(action.shape)
        # print(probs.log_prob(action).shape)
        return action, probs.log_prob(action), -probs.log_prob(action).mean(dim=0), values


def get_data_from_env(normalizer = (torch.tensor(1),torch.tensor(1))):
    ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
    local_observation = []
    local_action = []
    local_logprob = []
    local_reward = []
    local_timestep = []
    
    observation = env.get_obs()[0]
    previous_action = np.zeros((action_space))
    observation = np.hstack([observation,previous_action])
    observation = (observation-normalizer[1].numpy())/normalizer[0].numpy()**0.5
    
    local_observation.append(torch.Tensor(observation))
    timestep = np.ones((number_of_envs))
    local_timestep.append(torch.Tensor(timestep.copy()))
    for i in range(data_size) :
        
        # act and get observation 
        action, logprob = get_actor_critic_action_and_values(torch.Tensor(observation).to(device))
        action, logprob = action.cpu(), logprob.cpu()
        local_action.append(torch.Tensor(action))
        local_logprob.append(torch.Tensor(logprob))
        env.step(action)
        env.sim()
        observation, reward, info= env.get_obs()
        
        previous_action = action.squeeze()
        observation = np.hstack([observation,previous_action])
        observation = (observation-normalizer[1].numpy())/normalizer[0].numpy()**0.5
        terminated,truncated = False, info

        local_reward.append(torch.Tensor(reward))
        local_observation.append(torch.Tensor(observation))
        local_timestep.append(torch.Tensor(timestep.copy()))
        t.sleep(1./240.)
        timestep = (1 + timestep)*(1-(terminated | truncated))
    return local_observation, local_action, local_logprob, local_reward, local_timestep

class custom_dataset(Dataset):
    
    def __init__(self,data,data_size,number_of_envs,gamma):
        self.data_size = data_size
        self.number_of_envs = number_of_envs
        self.gamma = gamma
        self.obs, self.action, self.logprob, self.reward, self.timestep = data        
        self.local_return = [0 for i in range(data_size)]
        self.local_return = torch.hstack(self.get_G()).view(-1,1)
        self.local_observation = torch.vstack(self.obs)
        self.local_action = torch.vstack(self.action)
        self.local_logprob = torch.vstack(self.logprob).view(-1,1)
        self.local_reward = torch.hstack(self.reward).view(-1,1)
        # print(self.local_observation.shape)
        # print(self.local_action.shape)
        # print(self.local_logprob.shape)
        # print(self.local_return.shape)
        # print(self.local_reward.shape)

    def __len__(self):
        return self.data_size*self.number_of_envs
    
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

mlp = MLP().to(device)

### Normalize the return and obs
mlp.eval()
with torch.no_grad():
        data = get_data_from_env()
data = custom_dataset(data,data_size,number_of_envs,gamma)
obs_var_mean = torch.var_mean(data.local_observation,dim=0)

mlp.load_state_dict(torch.load(model_path,map_location=device))

for epoch in range(epochs):
    mlp = mlp.eval()
    # Sample data from the environment
    with torch.no_grad():
        data = get_data_from_env()
        

