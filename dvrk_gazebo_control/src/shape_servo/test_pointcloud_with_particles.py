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
from copy import copy, deepcopy
import rospy
# from dvrk_gazebo_control.srv import *
# from geometry_msgs.msg import PoseStamped, Pose
import open3d
# from utils import open3d_ros_helper as orh
# from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
#import pptk
# from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
# from dnn_architecture import Net, train
# import torch
# import torch.optim as optim
# from ShapeServo import *
from sklearn.decomposition import PCA
import timeit
from copy import deepcopy

ROBOT_Z_OFFSET = 0.1
angle_kuka_2 = -0.4
init_kuka_2 = 0.15



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


    pose_2 = gymapi.Transform()
    pose_2.p = gymapi.Vec3(0.0, 0.96, ROBOT_Z_OFFSET)
    pose_2.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002






    
    # Load soft objects' assets
    asset_root = "/home/baothach/sim_data/BigBird/BigBird_urdf_new" # Current directory
    soft_asset_file = "soft_box/soft_box.urdf"

    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0., 0.4, 0.03)
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.005    # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True
    # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)
        
    # create box asset
    box_size = 0.1
    box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)
    box_pose = gymapi.Transform()
    box_pose.p.x = 0.0
    box_pose.p.y = 0.4
    box_pose.p.z = 0.5 * box_size
 

    
    # set up the env grid
    # spacing = 0.75
    spacing = 0.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
  

    # cache some common handles for later use
    envs = []
    envs_obj = []
    kuka_handles = []
    kuka_handles_2 = []
    object_handles = []
    


    # create box asset
    box_size = 0.045
    box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)
    box_pose = gymapi.Transform()



    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    base_poses = []

    for i in range(num_envs):
    
        

        # add soft obj        
        env_obj = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs_obj.append(env_obj)        
        
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)
        # soft_actor = gym.create_actor(env_obj, box_asset, box_pose, "box", i, 0)
        # soft_actor = gym.create_actor(env_obj, syrup_asset, syrup_pose, "soft", i, 0)
        object_handles.append(soft_actor)

        # add box
        box_pose.p.x = 5
        box_pose.p.y =  5
        box_pose.p.z = 5 * box_size
        # box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
        box_handle = gym.create_actor(env_obj, box_asset, box_pose, "box", i, 0, segmentationId=1)


  



    # Camera setup
    cam_pos = gymapi.Vec3(1, 0.5, 1)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
    middle_env = envs_obj[num_envs // 2 + num_per_row // 2]
    gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    # Camera for point cloud setup
    cam_positions = []
    cam_targets = []
    cam_handles = []
    cam_width = 300
    cam_height = 300
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height
    cam_positions.append(gymapi.Vec3(0.2, 0.6, 0.2))
    cam_targets.append(gymapi.Vec3(0.0, 0.4, 0.05))
    # cam_positions.append(gymapi.Vec3(-0.5, 1.0, 0.5))
    # cam_targets.append(gymapi.Vec3(0.0, 0.4, 0.0))    

    for i, env_obj in enumerate(envs_obj):
        # for c in range(len(cam_positions)):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])





    final_points = []
    final_vtc_135 = []    
    final_vtc_30 = [] 

    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')



    # Some important paramters
    
    all_done = False
    state = "get point cloud initial"


    first_time_step = True 
    frame_count = 0
    frame_count_pc = 0



    start_time = timeit.default_timer()
    while (not gym.query_viewer_has_closed(viewer)) and (not all_done):

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        t = gym.get_sim_time(sim)
 

            
        if state == "get point cloud initial":
            rospy.loginfo("**Current state: " + state)

            frame_count_pc += 1 
            if frame_count_pc == 1:
                gym.refresh_particle_state_tensor(sim)
                particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
                # particle_states_initial = particle_state_tensor
                # print(particle_states_initial.shape)
                # print(particle_states_initial)
                # print(np.linalg.norm(particle_states_initial))     

                particle_states_initial = particle_state_tensor.numpy()[:, :3]
                # print(particle_states_initial.shape)
                # print(particle_states_initial)
                # print(np.linalg.norm(particle_states_initial))
                
                # Get feature vector for initial position
                pcd_goal = open3d.geometry.PointCloud()
                pcd_goal.points = open3d.utility.Vector3dVector(particle_states_initial)            
                state = "get point cloud"

        
        if state == "get point cloud":
            rospy.loginfo("**Current state: " + state)
            gym.refresh_particle_state_tensor(sim)
            particle_state_tensor = gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))
            particle_states = particle_state_tensor.numpy()[:, :3]
            state = "get feature vector from VFH and PCA"

        if state == "get feature vector from VFH and PCA":
            rospy.loginfo("**Current state: " + state)
            # Get feature vector
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(particle_states)            


            state = "get desired velocity and deform object"

        if state == "get desired velocity and deform object":
            rospy.loginfo("**Current state: " + state)          
            
          
            dist = np.linalg.norm(particle_states - particle_states_initial)

            print("***distance: ", dist) 






            state = "rest"

        if state == "rest":
            rospy.loginfo("**Current state: " + state)
            frame_count += 1
            if (frame_count % 100) == 0:
                state = "get point cloud"


   
            
        # step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    visualization_data = {"points": final_points}
    # with open('/home/baothach/shape_servo_data/uncertainty/uncertainty_vis_particles_no_touching', 'wb') as handle:
    #     pickle.dump(visualization_data, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

