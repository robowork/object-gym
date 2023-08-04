**Object-Gym Documentation**

![gym_envs](https://github.com/robowork/object-gym/assets/136655541/c6e4c029-2174-420f-b3a6-9e734007b4d0)
This repository contains the environment to train a policy to move large ungraspable objects by applying forces in a physics simulator. The simulation and training is massively parallelized, running end-to-end on the GPU. Unlike other Gym examples, the action space of this environment consists of applying forces at specific agent positions, and does not actuate the DOF. 


Required: 

- Python 3.8+
- PyTorch (1.10 recommended)
- CUDA (11.8 recommended)

\* Different versions of CUDA can be required based on which Nvidia driver is installed. This experiment was ran on the RTX 4090, which required CUDA 11.8. Older versions should be able to use PyTorch 1.10 and CUDA 11.3

Installation

1. Create a new python virtual environment
2. Install PyTorch 1.10:


    pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

3. Install debian version of CUDA 11.8 [here](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/)

- Ensure .bashrc is using correct version: `export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}`

4. Install Isaac Gym Preview 4 

- `cd isaacgym/python && pip install -e .`
- Check to see if it is running correctly


    cd examples && python 1080_balls_of_solitude.py

5. Clone this repository


    cd object-gym && pip install -e .

Usage

1. To train the policy with the default environment, run python train.py

- Command-line flags can be added, detailed in training script
- Each policy is saved in Results folder

2. To play a trained policy, run python play.py POLICY=\*file name\*

- The last training session is the default if no policy is specified

Troubleshooting

    NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

This error can be attributed to having SecureBoot running on your system, which prevents driver communication as a safety precaution. Restart your computer, and navigate to BIOS Setup -> Boot Options and disable SecureBoot

    RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)

This error comes from having the wrong CUDA version installed for your device. Check compatibility matrix and make sure .bashrc is using the correct version.

**Validation**

Validation for this project will be if the network can train an efficient policy in a manner of minutes, showcasing the parallelism of the GPU. Behavior can be determined to be efficient if it can happen quickly, using a realistic amount of force, and minimizes work done. This environment should be able to easily load in assets of varying shapes, and apply forces in an intelligent manner to move them to the desired location.

