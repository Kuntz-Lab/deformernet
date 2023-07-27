#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import

import sys
sys.path.append('/home/baothach/dvrk_grasp_pipeline_issac/src/dvrk_env/dvrk_gazebo_control/src')
import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from copy import copy
import rospy
from dvrk_gazebo_control.srv import *
from geometry_msgs.msg import PoseStamped, Pose
import GraspDataCollectionClient as dc_class
import open3d
from utils import open3d_ros_helper as orh
from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
#import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped

ROBOT_Z_OFFSET = 0.25


def check_reach_desired_position(i, desired_position, error = 0.01 ):
    '''
    Check if the robot has reached the desired goal positions
    '''
    current_position = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
    current_position = [x[0] for x in current_position]
    
    # absolute(a - b) <= (atol + rtol * absolute(b)) will return True
    # rtol: relative tolerance; atol: absolute tolerance 
    return np.allclose(current_position, desired_position, rtol=0, atol=error)

def get_current_joint_states(i):
    current_position = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
    current_position = [x[0] for x in current_position]
    return list(current_position)



if __name__ == "__main__":

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--object_type", "type": int, "default": 0, "help": "Type of bjects to place in the bin: 0 - box, 1 - meat can, 2 - banana, 3 - mug, 4 - brick, 5 - random"}])

    num_envs = args.num_envs
    


    # configure sim
    sim_type = args.physics_engine
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    if sim_type is gymapi.SIM_FLEX:
        sim_params.substeps = 4
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 6
        sim_params.flex.num_inner_iterations = 50
        sim_params.flex.relaxation = 0.7
        sim_params.flex.warm_start = 0.1
        sim_params.flex.shape_collision_distance = 5e-4
        sim_params.flex.contact_regularization = 1.0e-6
        sim_params.flex.shape_collision_margin = 1.0e-4
        sim_params.flex.deterministic_mode = True

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)



    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up ground
    gym.add_ground(sim, plane_params)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    # load robot assets
    asset_root = "../../assets"

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    #pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002


    asset_root = "./src/dvrk_env"
    kuka_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"


    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = True

    if sim_type is gymapi.SIM_FLEX:
        asset_options.max_angular_velocity = 40.

    print("Loading asset '%s' from '%s'" % (kuka_asset_file, asset_root))
    kuka_asset = gym.load_asset(sim, asset_root, kuka_asset_file, asset_options)



    # create box asset
    box_size = 0.1
    box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)
    box_pose = gymapi.Transform()

    # Load cube asset
    load_options = gymapi.AssetOptions()
    load_options.fix_base_link = True
    load_options.disable_gravity = True  
    load_options.thickness = 0.005
    asset_root = "../"
    cube_asset_file = "sim_data/BigBird/cube_2.urdf"
    cube_asset = gym.load_asset(sim, asset_root, cube_asset_file, load_options)

    
    # Load soft objects' assets
    asset_root = "/home/baothach/sim_data/BigBird/BigBird_urdf_new" # Current directory
    soft_asset_file = "soft_box/soft_box.urdf"

    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0., 0.4, 0.03)
    # soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.005    # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True
    # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)
        
    
  
    
    # set up the env grid
    spacing = 0.75
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # use position drive for all dofs; override default stiffness and damping values
    dof_props = gym.get_asset_dof_properties(kuka_asset)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(1000.0)
    dof_props["damping"].fill(200.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)


    # get joint limits and ranges for kuka
    lower_limits = dof_props['lower']
    upper_limits = dof_props['upper']
    ranges = upper_limits - lower_limits
    mids = 0.5 * (upper_limits + lower_limits)
    #num_dofs = len(kuka_dof_props)


    # default dof states and position targets
    num_dofs = gym.get_asset_dof_count(kuka_asset)
    default_dof_pos = np.zeros(num_dofs, dtype=np.float32)
    default_dof_pos = upper_limits

    default_dof_state = np.zeros(num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    # cache some common handles for later use
    envs = []
    kuka_handles = []
    object_handles = []
    






    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    base_poses = []

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)


        # add kuka
        kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", i, 1, segmentationId=11)
        
        # # Add object:
        # cube_handle = gym.create_actor(env, cube_asset, gymapi.Transform(p=gymapi.Vec3(0, 0.3, 0.1)), 'cube', i, 0)
        # object_handles.append(cube_handle)

        # # add soft obj        
        # soft_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 0, segmentationId=11)
        # object_handles.append(soft_actor)

        # # Test add bigbird obj        
        # syrup_actor = gym.create_actor(env, syrup_asset, syrup_pose, "soft", i, 0, segmentationId=11)
        # object_handles.append(syrup_actor)


        kuka_handles.append(kuka_handle)

    # Camera setup
    cam_pos = gymapi.Vec3(1, 0.5, 1)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
    middle_env = envs[num_envs // 2 + num_per_row // 2]
    gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    # Camera for point cloud setup
    cam_positions = []
    cam_targets = []
    cam_handles = []
    cam_width = 700
    cam_height = 700
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height
    cam_positions.append(gymapi.Vec3(0.5, 1.0, 0.5))
    cam_targets.append(gymapi.Vec3(0.0, 0.4, 0.0))   

    for i, env in enumerate(envs):
        # for c in range(len(cam_positions)):
            cam_handles.append(gym.create_camera_sensor(env, cam_props))
            gym.set_camera_location(cam_handles[i], env, cam_positions[0], cam_targets[0])


    tf = gym.get_camera_transform(sim,envs[0], cam_handles[0])
    print(tf.p)
    print(tf.r)





    
    
    
    
    

