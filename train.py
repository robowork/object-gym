from ppo import PPO
from env import BoxEnv
import torch
import random
import argparse
import sys
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()

parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
parser.add_argument('--num_envs', default=256, type=int)

args = parser.parse_args()

torch.manual_seed(19485)

policy = PPO(args)

def save_model(filename = "trained_model.pth"):
        torch.save(policy.net.state_dict(), filename)
        print("model saved as ", filename)

while True:
    try:
        policy.run()
    except KeyboardInterrupt:
        save = input("Would you like to save this policy? (y/n): ")
        if save == 'y':
            save_model("trained_model.pth")
        else: 
            plot = input("Would you like to plot results? (y/n): ")
            if plot == 'y':
                # variables for plotting
                steps = list(range(len(policy.env.distance_avg)))
                episode = [steps[i:i+(policy.env.max_episode_length-1)] for i in range(0, len(steps), policy.env.max_episode_length - 1)]
                distance_per_episode = [policy.env.box_y_avg[i:i+(policy.env.max_episode_length-1)] for i in range(0, len(steps), (policy.env.max_episode_length - 1))]
                reward_per_episode = [policy.env.reward_avg[i:i+(policy.env.max_episode_length-1)] for i in range(0, len(steps), (policy.env.max_episode_length - 1))]
                force_per_episode = [policy.env.force_avg[i:i+(policy.env.max_episode_length-1)] for i in range(0, len(steps), (policy.env.max_episode_length - 1))]

                plt.figure(1)
                for i in range(len(episode)):
                     if i%3 == 0:
                        if len(distance_per_episode[i]) == len(distance_per_episode[0]):
                            str_num = str(i+1)
                            distance_use = distance_per_episode[i]
                            distance_use[0] = 0
                            plt.plot(episode[0], distance_per_episode[i], label="Run " + str_num)
                plt.title("Steps v. Distance")
                plt.plot([0, 275], [4, 4], 'r--', lw=2)
                plt.text(100, 4.1, "Target Distance")
                plt.xlabel("Steps")
                plt.ylabel("Box Position (m)")
                plt.legend(loc='lower right')
                plt.grid()

                plt.figure(2)
                for i in range(len(episode)):
                     if i%3 == 0:
                        if len(reward_per_episode[i]) == len(reward_per_episode[0]):
                           str_num = str(i+1)
                           reward_use = reward_per_episode[i]
                           reward_use[0] = reward_use[1]
                           plt.plot(episode[0], reward_per_episode[i], label="Run " + str_num)
                plt.title("Steps v. Average Reward")
                plt.xlabel("Steps")
                plt.ylabel("Average Reward")
                plt.legend(loc='lower right')
                plt.grid()

                plt.show()
            
                sys.exit()
            else: 
                sys.exit()








