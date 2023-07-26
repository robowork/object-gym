from ppo import PPO
from env import BoxEnv
import torch
import random
import argparse
import sys

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
            sys.exit()








