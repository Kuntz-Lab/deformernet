# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import torch

from rlgpu.utils.torch_jit_utils import *
# from rlgpu.tasks.base.base_task import BaseTask
from tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
import math
from tasks.task_utils import *
import pickle
import rospy

import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
pkg_path = '/home/baothach/dvrk_shape_servo/src/dvrk_env/dvrk_gazebo_control'
sys.path.append(pkg_path + '/src')
from core import Robot
from behaviors import MoveToPose, TaskVelocityControl

sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *


ROBOT_Z_OFFSET = 0.25
two_robot_offset = 0.86
class ShapeServo(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
       
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.chamfer_success_threshold = 0.30
        self.chamfer_fail_threshold = 2.5
        self.num_points = 1024

        with open('/home/baothach/shape_servo_data/generalization/surgical_setup/goal_data/sample 1.pickle', 'rb') as handle:
            data = pickle.load(handle)
            self.goal_pc = data["partial pc"]
            self.pcd_goal = open3d.geometry.PointCloud()
            self.pcd_goal.points = open3d.utility.Vector3dVector(self.goal_pc) 

        with open('/home/baothach/shape_servo_data/RL_shapeservo/saved_init_states/box_init_states.pickle', 'rb') as handle:
            data = pickle.load(handle)
            self.saved_obj_state = data["saved_obj_state"]
            self.saved_robot_state = data["saved_robot_state"]
            self.saved_frame_state = data["saved_frame_state"]
    

        num_obs = 25
        num_acts = 3

        self.cfg["env"]["numObservations"] = self.num_points*3
        self.cfg["env"]["numActions"] = 3

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

 
        # create some wrapper tensors for different slices
        self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.469, 0.035, 0.035], device=self.device)
        # self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = torch.randn(1, 10, 2).to(self.device)
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]


 

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.mtp_behavior = None

        print("sim_params:", self.sim_params.flex.num_inner_iterations)
        self.count = 0

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # dvrk asset
        dvrk_pose = gymapi.Transform()
        dvrk_pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
        dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        # asset_options.thickness = 0.0001
        asset_options.thickness = 0.001

        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.max_angular_velocity = 40000.

        asset_root = "/home/baothach/dvrk_shape_servo/src/dvrk_env"
        dvrk_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"
        print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
        dvrk_asset = self.gym.load_asset(self.sim, asset_root, dvrk_asset_file, asset_options)

        # Soft object asset
        asset_root = "/home/baothach/sim_data/BigBird/BigBird_urdf_new"
        asset_root = "/home/baothach/sim_data/Custom/Custom_urdf"
        # soft_asset_file = "cheez_it_white_cheddar.urdf"
        soft_asset_file = "box.urdf"
        soft_pose = gymapi.Transform()
        # soft_pose.p = gymapi.Vec3(0.0, 0.50-two_robot_offset, 0.03)
        soft_pose.p = gymapi.Vec3(0.0, -0.42, 0.01818)
        soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
        # soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
        soft_thickness = 0.0005 
        # soft_thickness = 0.005

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.thickness = soft_thickness
        asset_options.disable_gravity = True
        soft_asset = self.gym.load_asset(self.sim, asset_root, soft_asset_file, asset_options)


        # # set franka dof properties
        dof_props = self.gym.get_asset_dof_properties(dvrk_asset)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"].fill(200.0)
        dof_props["damping"].fill(40.0)
        dof_props["stiffness"][8:].fill(1)
        dof_props["damping"][8:].fill(2)          

        # dof_props['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
        # dof_props["stiffness"][:8].fill(0.0)
        # dof_props["damping"][:8].fill(40.0)

        self.vel_limits = dof_props['velocity']
        # print("====vel limits:", self.vel_limits)
   
        # set up the env grid
        spacing = 0.0
        num_envs = 1
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(math.sqrt(num_envs))


        self.dvrks = []
        self.envs = []
        self.envs_obj = []
        self.object_handles = []
        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)



            dvrk_handle = self.gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i, 1, segmentationId=11)
            self.gym.set_actor_dof_properties(env, dvrk_handle, dof_props)

            # add soft obj        
            # env_obj = gym.create_env(sim, env_lower, env_upper, num_per_row)
            env_obj = env
         
            
            soft_actor = self.gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)
            
            self.object_handles.append(soft_actor)                  
            self.envs.append(env)
            self.envs_obj.append(env_obj)
            self.dvrks.append(dvrk_handle)        

        self.robot = Robot(self.gym, self.sim, self.envs[0], self.dvrks[0])
        
        cam_width = 256
        cam_height = 256    
        cam_pos = gymapi.Vec3(0.1, -0.5, 0.2)
        cam_target = gymapi.Vec3(0.0, -0.45, 0.00)    
        self.cam_handle, self.cam_prop = setup_cam(self.gym, self.envs[0], cam_width, cam_height, cam_pos, cam_target)
        
        self.reset()
        


    def compute_reward(self):
        contacts = [contact[4] for contact in self.gym.get_soft_contacts(self.sim)]
        lose_contact = not(9 in contacts and 10 in contacts)
        # print("===lose contacts: ", lose_contact)
        
        self.rew_buf[:], self.reset_buf[:] = compute_shape_servo_reward(
            self.reset_buf, self.pc, self.progress_buf, self.max_episode_length, self.pcd_goal,
            self.chamfer_success_threshold, self.chamfer_fail_threshold, lose_contact
        )

    def compute_observations(self):
        # print("==== Computing obs and rew")
        # self.count += 1
        # print("===",self.count)

        # self.obs_buf = torch.randn(1, 25)
        # self.obs_buf = torch.randn(1, self.num_points*3)
        # self.obs_buf = torch.from_numpy(get_point_cloud())
        # self.obs_buf = torch.from_numpy(get_partial_point_cloud(self.gym, self.sim, self.envs[0], self.cam_handle, self.cam_prop))
        self.pc = get_partial_point_cloud(self.gym, self.sim, self.envs[0], self.cam_handle, self.cam_prop)
        farthest_indices,_ = farthest_point_sampling(self.pc, self.num_points)
        self.obs_buf = torch.from_numpy(self.pc[farthest_indices.squeeze()]).permute(1,0).flatten().unsqueeze(0)
        # print("=====self.obs_buf.shape",torch.from_numpy(self.pc[farthest_indices.squeeze()]).permute(1,0).shape)

        return self.obs_buf.to(self.device)

    def pre_physics_step(self, actions, num_transitions):
        
        if num_transitions % 300 == 0:
            print("num_transitions", num_transitions)
        # print("=========actions",actions)

        # clamped_actions = np.clip(actions.squeeze().cpu().numpy(), [-0.1, 0, 0], 0.1)
        # clamped_actions = np.array([0.05,0.05,0.05])
        # print("=========clamped_actions",clamped_actions)
        # tvc_behavior = TaskVelocityControl(clamped_actions, self.robot, self.sim_params.dt, 3, self.vel_limits, self.init_ee_pose) 
        # action = tvc_behavior.get_action()

        # if action is not None:
        #     self.gym.set_actor_dof_velocity_targets(self.robot.env_handle, self.robot.robot_handle, action.get_joint_position()) 
        # print("complete:", tvc_behavior.is_complete_success()) 


        if num_transitions % 30 == 0 or self.mtp_behavior is None:
            # print("=========actions",actions)
            # new_pose = [-self.init_ee_pose[0] + clamped_actions[0], -self.init_ee_pose[1] + clamped_actions[1], 
            #             self.init_ee_pose[2] - ROBOT_Z_OFFSET + clamped_actions[2]]
            act = actions.squeeze().cpu().numpy()/30
            # if self.count <= 2:
            #     act = [0.08, 0.08, 0.00]
            # self.count += 1
            print("=========actions",act)
            ee_pose = self.robot.get_ee_cartesian_position() 
            new_pose = [-ee_pose[0] + act[0], -ee_pose[1] + act[1], 
                        ee_pose[2] - ROBOT_Z_OFFSET + act[2]]       # Robot frame      
            new_pose = np.clip(new_pose, self.low_ee_lim, self.high_ee_lim) #clamp to lims

            target_pose = list(new_pose) + [0, 0.707107, 0.707107, 0]            
            self.mtp_behavior = MoveToPose(target_pose, self.robot, self.sim_params.dt, 10*self.sim_params.dt, open_gripper=False)
        if not self.mtp_behavior.is_complete_failure():
            action = self.mtp_behavior.get_action()
            if action is not None:
                self.gym.set_actor_dof_position_targets(self.robot.env_handle, self.robot.robot_handle, action.get_joint_position())      
        else:
            self.mtp_behavior = None
        # print("complete:", self.mtp_behavior.is_complete_success())     
                              

    def post_physics_step(self, compute_rew_obs=True):        
        if self.reset_buf[0] == 1:
            self.reset()
        if compute_rew_obs:
            self.progress_buf += 1
            self.compute_observations()
            self.compute_reward()
        # print("===",self.progress_buf)

    def reset(self):
        # davinci_dof_states = self.gym.get_actor_dof_states(self.envs[0], self.dvrks[0], gymapi.STATE_NONE)
        # davinci_dof_states['pos'][4] = 0.2
        # davinci_dof_states['pos'][8] = 1.5
        # davinci_dof_states['pos'][9] = 0.8
        # self.gym.set_actor_dof_states(self.envs[0], self.dvrks[0], davinci_dof_states, gymapi.STATE_POS)  
        print("========================RESETING===========================")

        self.gym.set_actor_rigid_body_states(self.envs[0], self.dvrks[0], self.saved_robot_state, gymapi.STATE_ALL) 
        self.gym.set_particle_state_tensor(self.sim, gymtorch.unwrap_tensor(self.saved_obj_state))        

        self.gym.set_joint_target_position(self.envs[0], self.gym.get_joint_handle(self.envs[0], "dvrk", "psm_tool_gripper1_joint"), 0.35)
        self.gym.set_joint_target_position(self.envs[0], self.gym.get_joint_handle(self.envs[0], "dvrk", "psm_tool_gripper2_joint"), -0.35) 

        self.init_ee_pose = self.robot.get_ee_cartesian_position() 
        self.init_ee_pose[:2] = -self.init_ee_pose[:2]  # Robot frame
        self.init_ee_pose[2] -= ROBOT_Z_OFFSET

        self.low_ee_lim =  self.init_ee_pose[:3] + np.array([-0.1, 0, 0])
        self.high_ee_lim =  self.init_ee_pose[:3] + np.array([0.1, 0.1, 0.1])

        self.progress_buf[0] = 0
        self.reset_buf[0] = 0

#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def compute_shape_servo_reward(reset_buf, current_pc, progress_buf, max_episode_length, pcd_goal, 
    chamfer_success_threshold, chamfer_fail_threshold, lose_contact):
    # print("===  eps:", progress_buf[0])

    # if progress_buf >= max_episode_length - 1 or lose_contact:
    #     reset_buf = torch.tensor([1])

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(current_pc) 
    chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
    print("**Chamfer dist, eps:", chamfer_dist, progress_buf[0])
    # rewards = - torch.tensor([chamfer_dist])
    rewards = - torch.tensor([chamfer_dist])

    if chamfer_dist >= chamfer_fail_threshold or chamfer_dist <= chamfer_success_threshold or \
        progress_buf >= max_episode_length - 1 or lose_contact:
        reset_buf = torch.tensor([1])
    else:
        reset_buf = torch.tensor([0])


    # rewards = torch.tensor([10])
    # reset_buf = torch.tensor([0])
    # print("=======rewards shape:", rewards.shape)
    # print("=======reset_buf shape:", reset_buf.shape)


    return rewards, reset_buf

'''
paramters needed to change:
  chamfer_success_threshold
  noptepochs: 20
  nsteps
  nminibatches: 1 
  self.storage
  % something get new action (2 places)
  max eps length
  clean tensorboard
'''
