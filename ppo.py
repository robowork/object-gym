from env import BoxEnv

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal

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
    
class PPO:
    def __init__(self, args):
        self.args = args
        
        # initializes the environment
        self.env = BoxEnv(args)

        # hyperparamters for learning
        self.epoch = 5
        self.lr = 3e-4  # how often the nn updates weights
        self.gamma = 0.99
        self.lmbda = 0.95
        self.clip = 0.25
        self.rollout_size = 128
        self.chunk_size = 32
        self.mini_chunk_size = self.rollout_size // self.chunk_size
        self.mini_batch_size = self.args.num_envs * self.mini_chunk_size
        self.num_eval_freq = 100

        self.data = []
        self.score = 0
        self.run_step = 0
        self.optim_step = 0

        self.net = Net(self.env.num_obs, self.env.num_act).to(args.sim_device)

        # adds variance to the action space
        self.action_var = torch.full((self.env.num_act,), 0.1).to(args.sim_device)

        # optimizes the parameters of the nn
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def make_data(self):
        # organize the data/make batch
        data = []

        for _ in range(self.chunk_size):
            obs_lst, a_lst, r_lst, next_obs_lst, log_prob_lst, done_lst = [], [], [], [], [], []
            for _ in range(self.mini_chunk_size):
                # pops out the first entry of data and sets it to rollout
                rollout = self.data.pop(0)

                # all of those are contained in data, in that order
                obs, action, reward, next_obs, log_prob, done = rollout

                obs_lst.append(obs)
                a_lst.append(action)
                r_lst.append(reward.unsqueeze(-1))
                next_obs_lst.append(next_obs)
                log_prob_lst.append(log_prob)
                done_lst.append(done.unsqueeze(-1))

            obs, action, reward, next_obs, done = \
                torch.stack(obs_lst), torch.stack(a_lst), torch.stack(r_lst), torch.stack(next_obs_lst), torch.stack(done_lst)
            
            # compute the reward-to-go 
            with torch.no_grad():
                # calculates target as current reward + value of the next state (discounted by gamma)
                reward = reward.squeeze(-1)
                target = reward + self.gamma * self.net.v(next_obs) * done

                # delta is the difference between the what the expected value of the state is and the
                # actual value of the current observation, if delta > 0 then it was worse, delta < 0 means 
                # the value is more than thought

                # also called the TD error
                delta = target - self.net.v(obs)

            # compute the advantage of the action
            adv_list = []
            advantage = 0.0

            # maps the advantage of each action to show relative impact
            for delta_t in reversed(delta):
                advantage = self.gamma * self.lmbda * advantage + delta_t
                adv_list.insert(0, advantage)

            advantage = torch.stack(adv_list) 
            log_prob = torch.stack(log_prob_lst)

            mini_batch = (obs, action, log_prob, target, advantage)
            data.append(mini_batch)
        return data
    
    def update(self):
        # update actor/critic network
        data = self.make_data()

        for i in range(self.epoch):
            for mini_batch in data:
                obs, action, old_log_prob, target, advantage = mini_batch

                # pass the observations through actor network
                mu = self.net.pi(obs)

                # creates convariance matrix
                cov_mat = torch.diag(self.action_var)

                # creates guassian distribution with output of actor (mu) as mean and covariance matrix 
                dist = MultivariateNormal(mu, cov_mat)

                # returns log probability of an action
                log_prob = dist.log_prob(action)

                # returns log probability of an action
                ratio = torch.exp(log_prob - old_log_prob).unsqueeze(-1)

                # create the surrogate funcitons w/ clip
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage

                # loss is the minimum of the surrogates + loss from the critic network eval of state and target
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.net.v(obs), target)

                # zeros out the gradients for a clean slate
                self.optim.zero_grad()

                # initialize backprop on the mean of the gradients
                loss.mean().backward()
            
                # scale the gradients to prevent large policy updates
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)

                # perform single optimization step
                self.optim.step()

                self.optim_step += 1

    def run(self):
        # collect data
        # receive observations from env
        obs = self.env.obs_buf.clone()     # size(num_envs, num_obs)


        with torch.no_grad():

            # passes obs through actor net
            mu = self.net.pi(obs)

            # creates another covariance matrix
            cov_mat =  torch.diag(self.action_var)

            # calculates action distrubution using (mean, variance)
            dist = MultivariateNormal(mu, cov_mat)

            # samples action from a distribution
            action = dist.sample()

            # determines prob of policy taking that action
            log_prob = dist.log_prob(action)

            # clips the action from -1 to 1 (because of the cartpole env?)
            # the action space = num_envs
            action = action.clip(-1, 1)

        # sends action to env to simulate
        self.env.step(action)
        next_obs, reward, done = self.env.obs_buf.clone(), self.env.reward_buf.clone(), self.env.reset_buf.clone()
        self.env.reset()

        self.data.append((obs, action, reward, next_obs, log_prob, 1 - done))

        self.score += torch.mean(reward.float()).item() / self.num_eval_freq

        self.action_var = torch.max(0.01 * torch.ones_like(self.action_var), self.action_var - 0.00002)

        # training mode
        if len(self.data) == self.rollout_size:
            self.update()

        # evaluation mode
        if self.run_step % self.num_eval_freq == 0:
            print('Steps: {:04d} | Reward {:.04f} | Action Var {:.04f}'
                  .format(self.run_step, self.score, self.action_var[0].item()))
            self.score = 0


        self.run_step += 1

    





