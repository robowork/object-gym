from isaacgym import gymapi, gymtorch
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class BoxEnv:
    def __init__(self, args):
        self.args = args

        # configure sim 
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1 / 60.
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True

        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.use_gpu = True

        # task-specific params
        self.num_obs = 2 # current y-position of box, linear velocity in y-direction
        self.num_act = 1 # move along y-axis
        self.target_dist = 4
        self.reset_dist = 6 
        self.max_episode_length = 275
        
        # ----allocate buffers for storing events----

        # observations buffer: 2D tensor meant to hold all observations in an environment
        self.obs_buf = torch.zeros((self.args.num_envs, self.num_obs), device=self.args.sim_device)
        self.reward_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device)
        self.reset_buf = torch.ones(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        self.max_push_effort = 40 # newtons
        self.contact_time = 1 / 60 # seconds, how long the agent applies force 
        self.box_center_height = 0.25


        # variables for plotting graphs later
        self.force_avg = []
        self.reward_avg = []
        self.distance_avg = []
        self.box_y_avg = []
        self.force_plot = []

        # get gym interface
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        
        # create environments
        self.envs = self.create_envs()

        # generate viewer
        self.viewer = self.create_viewer()

        # step simulation
        self.gym.prepare_sim(self.sim)

        # get box position, coming from the root state sensor
        self.box_y_position, self.box_position, self.box_y_vel, self.root_states = self.get_states_tensor()


    #--------------------beginning of functions--------------# 

    def create_envs(self):
        # make ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = 0.5    
        plane_params.dynamic_friction = 0.5      
        plane_params.restitution = 0       
        plane_params.normal = gymapi.Vec3(0, 0, 1)

        self.gym.add_ground(self.sim, plane_params)

        # define env in space
        envs_per_row = int(np.sqrt(self.args.num_envs))

        # creates individual environment geometry
        spacing = 8
        lower = gymapi.Vec3(0, 0, 0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # add box asset
        asset_root = "urdf"
        asset_file = "cube.urdf"
        asset_opts = gymapi.AssetOptions()

        box_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_opts)

        pose = gymapi.Transform()

        # set up initial pose of box (at origin with no rotation)
        pose.p = gymapi.Vec3(0,0,self.box_center_height)
        pose.r = gymapi.Quat(1, 0.0, 0.0, 0.0)

        # generate envs
        envs = []
        box_handles = []
        
        print(f'Creating {self.args.num_envs} environments')
        for i in range(self.args.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, envs_per_row)

            box_handle = self.gym.create_actor(env, box_asset, pose, "box", i, 1, 0)
            box_handles.append(box_handle)

            self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0,0,1))

            envs.append(env)
        return envs



    def create_viewer(self):
        # create viewer looking at center of env
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(10, 0.0, 5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)
        return viewer


    def get_states_tensor(self):
        # get root state tensor of box
        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)

        # root states are size (num_actors, 13), represent (pos (x y z), rot (x y z w), vel (x y z), ang vel (x y z))
        root_states = gymtorch.wrap_tensor(_root_states)  

        # grab info from root states
        box_position = root_states[:,0:3] 
        box_y_position = box_position[:,1:2]
        box_y_vel = root_states[:,8:9]        

        return box_y_position, box_position, box_y_vel, root_states


    def get_obs(self, env_ids = None):
        # get an observation from each env

        # create a label from each env
        if env_ids is None:
            env_ids = torch.arange(self.args.num_envs, device=self.args.sim_device)

        self.gym.refresh_actor_root_state_tensor(self.sim)

        # creates observation for each env
        self.obs_buf[env_ids,0:1] = self.box_y_position[env_ids]
        self.obs_buf[env_ids,1:2] = self.box_y_vel[env_ids]


    def simulate(self):
         # step the physics
         self.gym.simulate(self.sim)
         self.gym.fetch_results(self.sim, True)

         # creates the target visualization at 4m away from each cube
         for i in range(len(self.envs)):
                self.gym.add_lines(self.viewer, self.envs[i], 1, (0,self.target_dist,0,0,self.target_dist,2), (1,0.5,0.5))

    def render(self):
         # update the viewer
         self.gym.step_graphics(self.sim)
         self.gym.draw_viewer(self.viewer, self.sim, True)
         self.gym.sync_frame_time(self.sim)

    def compute_reward(self, obs_buf, reset_dist, reset_buf, progress_buf, max_episode_length, actions_tensor):

        box_y, box_y_velo = torch.split(obs_buf, [1,1], dim=1)

        # creates target tensor
        target_y = torch.full((self.args.num_envs, 1), self.target_dist, device=self.args.sim_device)

        distance = torch.abs(box_y - target_y)

        # original reward to incentivize forward motion
        reward = box_y - (0.05*box_y_velo**2) - (0.4 * self.force_plot)

        # punish for going past the target
        reward = torch.where(box_y > target_y+0.25, reward - 8*(distance), reward)

        # conditional reward for being close to target and slow
        reward = torch.where((torch.logical_and(distance < 0.3, torch.abs(box_y_velo) < 1)), reward + 1, reward)
        reward = torch.where((torch.logical_and(distance < 0.2, torch.abs(box_y_velo) < 0.5)), reward + 2, reward)
        reward = torch.where((torch.logical_and(distance < 0.1, torch.abs(box_y_velo) < 0.2)), reward + 5, reward)

        # for plotting training plots
        self.reward_avg.append(float(torch.mean(reward)))
        self.distance_avg.append(float(torch.mean(torch.abs(distance))))
        self.force_avg.append(float(torch.mean(torch.abs(actions_tensor))))
        self.box_y_avg.append(float(torch.mean(box_y)))

        distance = distance.squeeze(-1)

        # if distance > reset_dist, places a 1 into the reset buffer to represent reset
        reset = torch.where(torch.abs(distance) > reset_dist, torch.ones_like(reset_buf), reset_buf)
        reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
        return reward, reset

    def step(self, actions):
        actions_tensor = torch.zeros((self.args.num_envs, 3), device=self.args.sim_device)

        # grabs actions from the PPO script (which are from -1 to 1), and multiplies it by a force scalar
        actions_tensor[:,1] = actions.squeeze(-1) * self.max_push_effort

        # for plotting
        self.force_plot = actions_tensor[:,1]

        # make contiguous
        self.force_pos = self.box_position.contiguous()

        self.apply_gradual_force(actions_tensor, self.force_pos)

        self.simulate()
        self.render()

        self.progress_buf += 1

        self.get_obs()
        self.reward_buf, self.reset_buf = self.compute_reward(self.obs_buf, self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length, actions_tensor)


    def reset(self):
        # collects env_ids to be reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) == 0:
            return
        
        reset_pos = torch.tensor([0,0,self.box_center_height], device=self.args.sim_device)
        reset_pos = reset_pos.repeat(len(env_ids), 1)
        
        self.box_position[env_ids, :] = reset_pos[:]

        # make sure each reset starts with a velocity of 0
        self.box_y_vel[env_ids, :] = 0

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # clear desired buffer states
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # get new obs after reset
        self.get_obs()


    def apply_gradual_force(self, actions_tensor, force_pos):
        # apply the force linearly over 1 second
        
        increment = int(self.contact_time * 60)

        force_increment = actions_tensor/increment

        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(force_increment), gymtorch.unwrap_tensor(force_pos), gymapi.ENV_SPACE)