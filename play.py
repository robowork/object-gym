from env import BoxEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.distributions import MultivariateNormal

parser = argparse.ArgumentParser()

parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
parser.add_argument('--num_envs', default=256, type=int)


args = parser.parse_args()

class Net(nn.Module):
    def __init__(self, num_obs=2, num_act=1):
        super(Net, self).__init__()

        # create a shared structre for actor/critic
        # sharing a network can increase performance in low-state systems
        self.shared_net = nn.Sequential(
            nn.Linear(num_obs, 256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LeakyReLU()
        )

        # set mean/variance of actor network
        # outputs action as a probability distribution
        self.to_mean = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_act),
            nn.Tanh()
        )

        # value for critic network
        # evaluates the value of being in a specific state

        self.to_value = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)     
        )

    # passes a value through the actor network
    def pi(self, x):
        x = self.shared_net(x)
        self.gamma = 0.99 
        mu = self.to_mean(x)
        return mu
    
    # passes a value through the critic network (state-value function)
    def v(self, x):
        x = self.shared_net(x)
        x = self.to_value(x)
        return x
    

# setup environment
env = BoxEnv(args)


def play_trained_policy(filename="trained_model.pth"):

    # initialize nn
    trained_model = Net(env.num_obs, env.num_act)

    # load trained weights into the net
    trained_model.load_state_dict(torch.load(filename, map_location=args.sim_device))

    # put into evaluation mode
    trained_model.eval()

    trained_model = trained_model.to(args.sim_device)

    obs = env.obs_buf.to(args.sim_device)     # size(num_envs, num_obs)

    action_var = torch.full((1,), 0.1).to(args.sim_device)


    with torch.no_grad():

        # passes obs through actor net
        mu = trained_model.pi(obs)

        # creates another covariance matrix
        cov_mat = torch.diag(action_var)

        # calculates action distrubution using (mean, variance)
        dist = MultivariateNormal(mu, cov_mat)

        # samples action from a distribution
        action = dist.sample()

        # determines prob of policy taking that action
        log_prob = dist.log_prob(action)

        # clips the action from -1 to 1
        # the action space = num_envs
        action = action.clip(-1, 1)

        # sends action to env to simulate
        env.step(action)

        # next_obs, reward, done = env.obs_buf.clone(), env.reward_buf.clone(), env.reset_buf.clone()
        env.reset()

x = True

while x == True:
    play_trained_policy("trained_model.pth")

