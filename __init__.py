import numpy as np
from torchrl.modules import TanhNormal
from torch.distributions import Normal
import seaborn as sns
import matplotlib.pyplot as plt
import torch
loc = torch.zeros((9,6))
scale = .2*torch.ones((9,6))
# print(loc)
# loc, scale = loc.T, scale.T
distr = TanhNormal(loc=loc,scale=scale,max=np.pi/4,min=-np.pi/4)
action  = distr.sample((100,))
print(torch.min(action))
print(torch.max(action))
# sns.kdeplot(data=action.squeeze())
print(action.shape)
print(distr.log_prob(action).flatten().shape)
entropy = -distr.log_prob(action)
print(entropy.shape)
# entropy_loss = entropy.mean()
# print(entropy_loss)
# loc = torch.zeros((5))
# scale = torch.ones((5))
# distr = Normal(loc=loc,scale=scale)
# action = distr.sample((1,))
# sns.kdeplot(data=action)
# print(action.shape)
# print(distr.log_prob(action).shape)
# print(distr.entropy())
plt.show()
# print(-distr.log_prob(action).mean(dim=0))
# print(distr.log_prob(action).mean(dim=0))
# import numpy as np
# import matplotlib.pyplot as plt
# num_robot = 16
# if num_robot/np.sqrt(num_robot)==int(np.sqrt(num_robot)):
#     pass
# else:
#     print('num_robot must be a prime')
# nrow = int(np.sqrt(num_robot))
# print(nrow)
# x = np.linspace(-(nrow+1)/2,(nrow+1)/2,nrow)
# print(x)
# xv,yv = np.meshgrid(x,x)
# xv, yv = np.hstack(xv), np.hstack(yv)
# zv = np.ones_like(xv)
# corr_list = np.vstack((xv,yv,zv)).transpose()
# print(corr_list)
# plt.scatter(xv,yv)
# plt.show()
# print(np.meshgrid([1,2],[3,4,],[5,6]))