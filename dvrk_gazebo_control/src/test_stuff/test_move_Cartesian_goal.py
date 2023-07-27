#!/usr/bin/env python3
"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Kuka bin perfromance test
-------------------------------
Test simulation perfromance and stability of the robotic arm dealing with a set of complex objects in a bin.
"""

from __future__ import print_function, division, absolute_import

import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from copy import copy
import rospy
from dvrk_gazebo_control.srv import *
from geometry_msgs.msg import PoseStamped, Pose
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped




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
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 40
    sim_params.flex.relaxation = 0.7
    sim_params.flex.warm_start = 0.1
    sim_params.flex.shape_collision_distance = 5e-6
    sim_params.flex.contact_regularization = 1.0e-6
    sim_params.flex.shape_collision_margin = 1.0e-4
elif sim_type is gymapi.SIM_PHYSX:
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 25
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.rest_offset = 0.001

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

# load assets
asset_root = "../../assets"

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.35)
#pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

asset_options = gymapi.AssetOptions()
asset_options.armature = 0.001
asset_options.fix_base_link = True
asset_options.thickness = 0.002


asset_root = "./src/dvrk_env"
#kuka_asset_file = "urdf/kuka_allegro_description/kuka_allegro.urdf"
#kuka_asset_file = "urdf/daVinci_description/robots/psm_from_WPI_test_2.urdf"  # change
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

# set up the env grid
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# use position drive for all dofs; override default stiffness and damping values
dof_props = gym.get_asset_dof_properties(kuka_asset)
dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
dof_props["stiffness"].fill(1000.0)
dof_props["damping"].fill(200.0)
# dof_props["stiffness"][:8].fill(1000.0)
# dof_props["damping"][:8].fill(200.0)
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






print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))
base_poses = []

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)


    # add kuka
    kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", i, 1)
    
    # add box
    box_pose.p.x = 0.0
    box_pose.p.y = 0.3
    box_pose.p.z = 0.5 * box_size
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0, segmentationId=1)    
 


    kuka_handles.append(kuka_handle)

# Camera setup
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)



# set dof properties
for env in envs:
    gym.set_actor_dof_properties(env, kuka_handles[i], dof_props)




# Open/close gripper
def move_gripper(i, open_gripper = True):
    pos_targets = np.zeros(num_dofs, dtype=np.float32)       
    current_position = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)        
    for j in range(num_dofs):
        pos_targets[j] = current_position[j][0]    
    if open_gripper:
        pos_targets[-1] = 0.8
        pos_targets[-2] = 0.8
    else:
        pos_targets[-1] = 0.1
        pos_targets[-2] = 0.1       
    
    gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets)    


def arm_moveit_planner_client(go_home=False, place_goal_pose=None, cartesian_pose=None):
    #bao123 
    '''
    return Is there any plan?
    calculate plan and assign to self.planning_response
    '''
    
    rospy.loginfo('Waiting for service moveit_cartesian_pose_planner.')
    rospy.wait_for_service('moveit_cartesian_pose_planner')
    rospy.loginfo('Calling service moveit_cartesian_pose_planner.')
    try:
        planning_proxy = rospy.ServiceProxy('moveit_cartesian_pose_planner', PalmGoalPoseWorld)
        planning_request = PalmGoalPoseWorldRequest()
        if go_home:
            planning_request.go_home = True
        elif place_goal_pose is not None:
            planning_request.palm_goal_pose_world = place_goal_pose
        else:
            # planning_request.palm_goal_pose_world = self.mount_desired_world.pose
            planning_request.palm_goal_pose_world = cartesian_pose
        planning_response = planning_proxy(planning_request) 
    except (rospy.ServiceException):
        rospy.loginfo('Service moveit_cartesian_pose_planner call failed')
    rospy.loginfo('Service moveit_cartesian_pose_planner is executed %s.'
            %str(planning_response.success))

    if not planning_response.success:
        rospy.loginfo('Does not have a plan to execute!')
    # print("-------------", planning_response.plan_traj)
    # Convert JointTrajectory message to list    
    plan_list = []
    for point in planning_response.plan_traj.points:
        plan_list.append(list(point.positions))
    return plan_list

def check_reach_desired_position(i, desired_position):
    '''
    Check if the robot has reached the desired goal positions
    '''
    current_position = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
    current_position = [x[0] for x in current_position]
    # print(current_position)
    # print(desired_position)
    
    # absolute(a - b) <= (atol + rtol * absolute(b)) will return True
    # rtol: relative tolerance; atol: absolute tolerance 
    return np.allclose(current_position, desired_position, rtol=0, atol=0.01)


def execute_traj(i, traj, open_gripper = True):
    '''
    execute a plan (trajectory) from MoveIt
    wait until the previous desired position has been reached
    Traj: list, each element is another list which contains joint angles    
    '''
    done = False
    traj_index = 0
    if open_gripper:
        g1_joint_value = 1
        g2_joint_value = 1
    else:
        g1_joint_value = 0
        g2_joint_value = 0

    while not done:
        pos_targets = np.array(traj[traj_index]+[g1_joint_value, g2_joint_value], dtype=np.float32) 
        gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets)  
        
        if check_reach_desired_position(i, pos_targets):
            traj_index += 1    
        
        if traj_index == len(traj):
            done = True    
    


rospy.init_node('isaac_grasp_client')
start_time = 1
frame_count = 0
get_traj_from_moveit = True
get_state = False
count = 0
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
  
    # check if we should start
    t = gym.get_sim_time(sim)
    
    for i in range(num_envs):  
        
        # 0. Test get pose of tool yaw link
        if get_state == True:
            if count == 60:
                state = gym.get_actor_rigid_body_states(env, kuka_handles[i], gymapi.STATE_POS)
                # for stuff in state["pose"]:
                #     print(stuff)
                #     print("-------------------------")
                print(to_PoseStamped(state[-3]))
                dof_state = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
                print("dof gripper: ", dof_state['pos'][8:])
                print(np.allclose(dof_state['pos'][8:], [0.5, 0.5], rtol=0, atol=0.001))
                print(np.allclose(dof_state['pos'][8:], [0.4, 0.4], rtol=0, atol=0.001))
                get_state = False
            count += 1
        # 1. Test check reach desired position
        # pos_targets = np.array([-0.3356692769522132, get_traj_from_moveitposition(i, pos_targets)
        # print(reach_desired_position)  

        # 2. Test reach a desired position:
        # if (t >= start_time):
            
        #     pos_targets = np.array([-0.3356692769522132, -0.18960693104623297, 0.9355599880218506, 0.045651170304702046,\
        #                              0.020980324428042426, 1.7311636090617426, -0.5123334395848442, -0.4493461734062968,\
        #                              1,1], dtype=np.float32)            
        #     gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets)

        # 3. Test execute trajectory from MoveIt (given joint angles)      
        # if get_traj_from_moveit:
        #     plan_traj = arm_moveit_planner_client(go_home=True, place_goal_pose=None)
        #     get_traj_from_moveit = False
        #     traj_index = 0
        #     done = False
        #     print(plan_traj)
        # plan_traj_with_gripper = [plan+[1,1] for plan in plan_traj]
        
        # if not done:
        #     pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
        #     gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets)        
        #     if check_reach_desired_position(i, pos_targets):
        #         traj_index += 1                
        #     if traj_index == len(plan_traj):
        #         done = True   



        # 4. Test execute trajectory from MoveIt ((given Cartesian pose) , also test add scene
        if get_traj_from_moveit:
         
            
            cartesian_pose = Pose()
            cartesian_pose.orientation.x = 0
            cartesian_pose.orientation.y = 0.707107
            cartesian_pose.orientation.z = 0.707107
            cartesian_pose.orientation.w = 0
            cartesian_pose.position.x = 0.15
            cartesian_pose.position.y = 0.3
            cartesian_pose.position.z = -0.3
     
            # cartesian_pose.orientation.x = 0.30794364345911074
            # cartesian_pose.orientation.y = 0.6158872869182215
            # cartesian_pose.orientation.z = 0
            # cartesian_pose.orientation.w = 0.7251576120166157
            # cartesian_pose.position.x = 0.0
            # cartesian_pose.position.y = 0.3
            # cartesian_pose.position.z = -0.3
            plan_traj = arm_moveit_planner_client(go_home=False, place_goal_pose=None, cartesian_pose=cartesian_pose)
            get_traj_from_moveit = False
            traj_index = 0
            done = False
            # print(plan_traj)
        plan_traj_with_gripper = [plan+[0.5,0.5] for plan in plan_traj]
        
        if not done:
            pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
            gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets)        
            if check_reach_desired_position(i, pos_targets):
                traj_index += 1                
            if traj_index == len(plan_traj):
                
                get_state = True
                done = True  
        


        # step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

state = gym.get_actor_rigid_body_states(env, kuka_handles[i], gymapi.STATE_POS)
for stuff in state["pose"]:
    print(stuff)
    print("-------------------------")
print(to_PoseStamped(state[-3]))
print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
