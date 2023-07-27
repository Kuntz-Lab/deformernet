#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')
import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from copy import copy, deepcopy
import rospy
# from dvrk_gazebo_control.srv import *
from geometry_msgs.msg import PoseStamped, Pose
from GraspDataCollectionClient import GraspDataCollectionClient
import open3d
from utils import open3d_ros_helper as orh
from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
#import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from utils.isaac_utils import fix_object_frame, get_pykdl_client
# from utils.record_data_h5 import RecordGraspData_sparse
import pickle
# from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
from PIL import Image
import transformations
import random

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl2



ROBOT_Z_OFFSET = 0.25
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 0.86



def init():
    for i in range(num_envs):
        # # Kuka 2
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][4] = 0.2
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        gym.set_actor_dof_states(envs[i], kuka_handles_2[i], davinci_dof_states, gymapi.STATE_POS)

def get_point_cloud():
    gym.refresh_particle_state_tensor(sim)
    particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
    point_cloud = particle_state_tensor.numpy()[:, :3]  
    
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(point_cloud))
    # open3d.visualization.draw_geometries([pcd])     
    # return list(point_cloud)
    return point_cloud.astype('float32')


def get_partial_point_cloud(i):

    # Render all of the image sensors only when we need their output here
    # rather than every frame.
    gym.render_all_camera_sensors(sim)

    points = []
    print("Converting Depth images to point clouds. Have patience...")
    # for c in range(len(cam_handles)):
    
    # print("Deprojecting from camera %d, %d" % i))
    # Retrieve depth and segmentation buffer
    depth_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_SEGMENTATION)


    # Get the camera view matrix and invert it to transform points from camera to world
    # space
    
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs_obj[i], cam_handles[0])))

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    # Ignore any points which originate from ground plane or empty space
    # depth_buffer[seg_buffer == 11] = -10001

    centerU = cam_width/2
    centerV = cam_height/2
    for k in range(cam_width):
        for t in range(cam_height):
            if depth_buffer[t, k] < -3:
                continue

            u = -(k-centerU)/(cam_width)  # image-space coordinate
            v = (t-centerV)/(cam_height)  # image-space coordinate
            d = depth_buffer[t, k]  # depth buffer value
            X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
            p2 = X2*vinv  # Inverse camera view to get world coordinates
            # print("p2:", p2)
            if p2[0, 2] > 0.01:
                points.append([p2[0, 0], p2[0, 1], p2[0, 2]])

    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(points))
    # open3d.visualization.draw_geometries([pcd]) 

    # return points
    return np.array(points).astype('float32')




if __name__ == "__main__":


    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--obj_type", "type": str, "default": None, "help": "hemis1k, hemis5k, etc."},
            {"name": "--prim_name", "type": str, "default": "hemis", "help": "Select primitive shape. Options: box, cylinder, hemis"},
            {"name": "--stiffness", "type": str, "default": "1k", "help": "Select object stiffness. Options: 1k, 5k, 10k"},
            {"name": "--obj_name", "type": int, "default": 0, "help": "select variations of a primitive shape"},            
            {"name": "--inside", "type": str, "default": "True", "help": "inside train distribution"},
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"}])

    num_envs = args.num_envs
    
    args.headless = args.headless == "True"
    args.inside = args.inside == "True"
    args.obj_type = f"{args.prim_name}_{args.stiffness}"
    args.obj_name = f"{args.prim_name}_{args.obj_name}"

    main_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_{args.obj_type}Pa/evaluate"
    objects_path = "/home/baothach/shape_servo_data/evaluation"



    # configure sim
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    if sim_type is gymapi.SIM_FLEX:
        sim_params.substeps = 4
        # print("=================sim_params.dt:", sim_params.dt)
        sim_params.dt = 1./60.
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 10#10
        sim_params.flex.num_inner_iterations = 50
        sim_params.flex.relaxation = 0.7
        sim_params.flex.warm_start = 0.1
        sim_params.flex.shape_collision_distance = 5e-4
        sim_params.flex.contact_regularization = 1.0e-6
        sim_params.flex.shape_collision_margin = 1.0e-4
        sim_params.flex.deterministic_mode = True

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)

    # Get primitive shape dictionary to know the dimension of the object   
    if args.inside:
        # object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "inside")
        object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "inside")
    else:
        object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "outside") 

    with open(os.path.join(object_meshes_path, args.obj_name + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)    
    r = data["radius"]
    h = data["height"]

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up ground
    gym.add_ground(sim, plane_params)

    # create viewer
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            quit()

    # load robot assets
    pose_2 = gymapi.Transform()
    pose_2.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    # pose_2.p = gymapi.Vec3(0.0, 0.85, ROBOT_Z_OFFSET)
    pose_2.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.001#0.0001


    asset_root = "./src/dvrk_env"
    kuka_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"


    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    if sim_type is gymapi.SIM_FLEX:
        asset_options.max_angular_velocity = 40000.

    print("Loading asset '%s' from '%s'" % (kuka_asset_file, asset_root))
    kuka_asset = gym.load_asset(sim, asset_root, kuka_asset_file, asset_options)

    if args.inside:
        # asset_root = os.path.join(objects_path, "urdf", args.obj_type, "inside_smaller")
        asset_root = os.path.join(objects_path, "urdf", args.obj_type, "inside")
    else:
        asset_root = os.path.join(objects_path, "urdf", args.obj_type, "outside") 


    soft_asset_file = args.obj_name + ".urdf"    
    # asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/test"
    # soft_asset_file = "long_cylinder.urdf"
    # asset_root = "/home/baothach/Downloads"
    # soft_asset_file = "test_cylinder.urdf"

    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0, 0.4-two_robot_offset, r/2.0)
    soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
    soft_thickness = 0.001#0.0005    # important to add some thickness to the soft body to avoid interpenetrations





    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)

        
    
 
    
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
    


    num_per_row = int(math.sqrt(num_envs))
    base_poses = []

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add kuka2
        kuka_2_handle = gym.create_actor(env, kuka_asset, pose_2, "kuka2", i, 1, segmentationId=11)        
        

        # add soft obj        
        env_obj = env
        env_obj = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs_obj.append(env_obj)        
        
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)
        object_handles.append(soft_actor)


        kuka_handles_2.append(kuka_2_handle)



    dof_props_2 = gym.get_asset_dof_properties(kuka_asset)
    dof_props_2["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props_2["stiffness"].fill(200.0)
    dof_props_2["damping"].fill(40.0)
    dof_props_2["stiffness"][8:].fill(1)
    dof_props_2["damping"][8:].fill(2)  
    vel_limits = dof_props_2['velocity']    
    print("======vel_limits:", vel_limits)
    
    

    # Camera setup
    if not args.headless:
        # cam_pos = gymapi.Vec3(1, 0.5, 1)
        # # cam_pos = gymapi.Vec3(0.3, -0.7, 0.3)
        # # cam_pos = gymapi.Vec3(0.3, -0.1, 0.5)  # final setup for thin layer tissue
        # # cam_pos = gymapi.Vec3(0.5, -0.36, 0.3)
        # # cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
        # cam_target = gymapi.Vec3(0.0, -0.36, 0.1)
        cam_pos = gymapi.Vec3(0.5, -0.8, 0.5)
        cam_target = gymapi.Vec3(0.0, -0.36, 0.1)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    # Camera for point cloud setup
    cam_positions = []
    cam_targets = []
    cam_handles = []
    cam_width = 256
    cam_height = 256
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height
    # cam_positions.append(gymapi.Vec3(0.12, -0.55, 0.15))
    cam_positions.append(gymapi.Vec3(0.17, -0.62, 0.2))
    cam_targets.append(gymapi.Vec3(0.0, 0.40-two_robot_offset, 0.01))
  

    
    for i, env_obj in enumerate(envs_obj):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])



    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

        

    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')
    rospy.logerr("======Loading object ... " + str(args.obj_name))  
    rospy.logerr(f"Object type ... {args.obj_type}; inside: {args.inside}")  
    # rospy.logerr(str(args.headless))
    # rospy.logerr(str(args.inside))

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    first_time = True
    

    if args.inside:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "inside")
    else:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "outside") 

    os.makedirs(goal_recording_path, exist_ok=True)

    goal_datas = []
    goal_point_count = 0
    max_goal_count = 10#10
    
    terminate_count = 0
    sample_count = 0
    frame_count = 0
    group_count = 0

    max_group_count = 10
    max_sample_count = 1

    dc_client = GraspDataCollectionClient()
    fail_mtp = False


    
    start_time = timeit.default_timer()   
    close_viewer = False
    robot = Robot(gym, sim, envs[0], kuka_handles_2[0])

    while (not close_viewer) and (not all_done): 



        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 

        if state == "home" :   
            frame_count += 1
            # gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.15)
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.203)            
            if frame_count == 10:
                rospy.loginfo("**Current state: " + state + ", current sample count: " + str(sample_count))

                if first_time:
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                    saved_frame_state = deepcopy(gym.get_actor_rigid_body_states(envs_obj[i], object_handles[i], gymapi.STATE_ALL))
                    first_time = False

                state = "generate preshape"
                
                
                frame_count = 0

                current_pc = get_point_cloud()
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                pc_ros_msg = fix_object_frame(pc_ros_msg)



        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
            print("==========h:", h)
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg, others=[h])               
            cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            target_pose = [-cartesian_goal.position.x, -cartesian_goal.position.y, cartesian_goal.position.z-ROBOT_Z_OFFSET-0.01,
                            0, 0.707107, 0.707107, 0]


            mtp_behavior = MoveToPose(target_pose, robot, sim_params.dt, 1)
            if mtp_behavior.is_complete_failure():
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "reset"  
                fail_mtp = True              
            else:
                rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                state = "move to preshape"
                # rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))
            pc_init = get_partial_point_cloud(i)
            full_pc_init = get_point_cloud()

        if state == "move to preshape":         
            action = mtp_behavior.get_action()

            if action is not None:
                gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())      
                        
            if mtp_behavior.is_complete():
                state = "grasp object"   
                rospy.loginfo("Succesfully executed PRESHAPE moveit arm plan. Let's fucking grasp it!!") 

        
        if state == "grasp object":             
            rospy.loginfo("**Current state: " + state)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), -2.5)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), -3.0)         

            g_1_pos = 0.35
            g_2_pos = -0.35
            dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)
            if dof_states['pos'][8] < 0.35:
                                       
                state = "get shape servo plan"
                    
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), g_1_pos)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), g_2_pos)         
                
                gym.refresh_particle_state_tensor(sim)
                saved_object_contact_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                saved_robot_contact_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))

                _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                init_eulers = transformations.euler_from_matrix(init_pose)

                dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                dof_props_2["stiffness"][:8].fill(0.0)
                dof_props_2["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)                

        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 
            print("==========h:", h)
            # max_x = max_y = max_z = min(h, 0.3) * 0.7 * 0.8 
            # max_x = max_y = max_z = min(h, 0.3) * 0.5 * 0.8 
            # max_x = max_x * 3/4
            # max_y = max_y * 3/4
            # max_z = max_z * 3/4

            # delta_x = np.random.uniform(low = -max_x, high = max_x)
            # max_x = 0.065
            # max_z = 0.05
            # max_y = 0.065
            # rospy.logerr(f"{max_y}")
            # delta_x = random.uniform(*random.choice([(-max_x, -max_x/2), (max_x/2, max_x)]))
            
            
            ####################################################
            max_x = 1 * 1/2 * 1/2*h
            max_y = 1 * 1/2 * 1/2*h
            max_z = h/0.45 * 0.08 #min(0.05, max_z)
            
            # delta_x = -max_x#np.random.choice([-max_x,max_x])
            # delta_y = max_y#np.random.uniform(low = max_y/1.5, high = max_y)
            # delta_z = max_z#np.random.uniform(low = 0.03, high = max_z)   

            delta_x = np.random.uniform(low = max_x/2, high = max_x)
            delta_x *= np.random.choice([-1,1])
            delta_y = np.random.uniform(low = max_y/2, high = max_y)
            delta_z = max_z

            # delta_alpha = np.random.uniform(low = -np.pi/3, high = np.pi/3)
            # delta_beta = np.random.uniform(low = -np.pi/3, high = np.pi/3) 
            # delta_gamma = np.random.uniform(low = -np.pi/2, high = np.pi/2)
            # delta_alpha = np.random.choice([-np.pi/3,np.pi/3]) #np.random.choice([-np.pi/3,np.pi/3])
            # delta_beta = np.random.choice([-np.pi/3,np.pi/3]) #np.random.choice([-np.pi/3,np.pi/3])
            # delta_gamma = np.random.choice([-np.pi/2,np.pi/2])

            delta_alpha, delta_beta = 0, 0
            # if delta_x <= 0:
            #     delta_alpha = np.random.uniform(low = 0*np.pi/4, high = np.pi/3)
            #     delta_beta = np.random.uniform(low = 0, high = np.pi/3)
            # else:
            #     delta_alpha = -np.random.uniform(low = 0*np.pi/4, high = np.pi/3)
            #     delta_beta = np.random.uniform(low = 0, high = np.pi/3)


            # delta_gamma = 0
            if delta_x <= 0:
                delta_gamma = -np.pi/2
            else:
                delta_gamma = np.pi/2


            # if h <= 0.25:
            #     max_y = h / 2 * 0.5 - 0.01
            #     delta_y = np.random.uniform(low = max_y/2, high = max_y)
            ####################################################
            # max_x = max_y = max_z = h * 0.5 * 0.8  #h * 0.7 * 0.8 
            # # max_z = h / 2.0

            # delta_x = np.random.uniform(low = -max_x, high = max_x)
            # delta_y = np.random.uniform(low = 0.0, high = max_y)
            # delta_z = np.random.uniform(low = 0.0, high = max_z)     
            # delta_alpha, delta_beta, delta_gamma = np.pi/300000, -np.pi/300000, np.pi/300000       



            # delta_alpha, delta_beta, delta_gamma = np.pi/300000, -np.pi/300000, -np.pi/2 
            # # # delta_x, delta_y, delta_z = -max_x/1,max_y*1.3,0.01#max_z
            # # delta_x, delta_y, delta_z = -max_x/1,max_y*1,0.05#max_z
            # # delta_x, delta_y, delta_z = -0.065/1.5,0.06/1.5,0.05#max_z
            # delta_x, delta_y, delta_z = -0.12/1,0.08/1,0.08#max_z
            # delta_x, delta_y, delta_z = -0.06/1,0.04/1,0.04#max_z

            print("select x, y, z, a, b, g:", delta_x, delta_y, delta_z, " | ", delta_alpha, delta_beta, delta_gamma)

            x = delta_x + init_pose[0,3]
            y = delta_y + init_pose[1,3]
            z = delta_z + init_pose[2,3]
            alpha = delta_alpha + init_eulers[0]
            beta = delta_beta + init_eulers[1]
            gamma = delta_gamma + init_eulers[2]

            tvc_behavior = TaskVelocityControl2([x,y,z,alpha,beta,gamma], robot, sim_params.dt, 3, vel_limits=vel_limits, pos_threshold = 1e-3, ori_threshold=5e-2)
            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            state = "move to goal"

        if state == "move to goal":
            main_ins_pos = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"))
            if main_ins_pos <= 0.042:
                rospy.logerr("Exceed joint constraint")
                state = "reset"
            
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
                rospy.logerr("Lost contact with robot")
                # all_done = True
                
                # group_count += 1
                state = "reset"


            else:
                if frame_count % 15 == 0:                   
                    terminate_count += 1
                
                if frame_count == 0:
                    curr_trans = get_pykdl_client(robot.get_arm_joint_positions())[1] 
                
                frame_count += 1           
                
                action = tvc_behavior.get_action() 
                if action is not None and gym.get_sim_time(sim) - closed_loop_start_time <= 5: 
                    gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())
                    # print("Still moving ...")
                else:   
                    rospy.loginfo("Succesfully executed moveit arm plan. Let's record point cloud!!")                
                    
                    _, final_trans = get_pykdl_client(robot.get_arm_joint_positions())
                    p, R = tvc_behavior.get_transform(curr_trans, final_trans)
                    mani_point = [-curr_trans[0,3], -curr_trans[1,3], curr_trans[2,3] + ROBOT_Z_OFFSET]                            


                    partial_goal_pcs = (pc_init, get_partial_point_cloud(i))
                    full_goal_pcs = (full_pc_init, get_point_cloud())                    
                    

                    goal_data = {"full pcs": full_goal_pcs, "partial pcs": partial_goal_pcs, \
                                "obj contact state": saved_object_contact_state, "robot contact state": saved_robot_contact_state, \
                                "pos": p, "rot": R, "mani_point": mani_point, "obj_name": args.obj_name,
                                "saved_object_state": saved_object_state}
                    
                    goal_datas.append(goal_data)
                    goal_point_count += 1
                    
                    frame_count = 0
                    terminate_count = 0
                    sample_count += 1
                    print("group ", group_count, ", sample ", sample_count)
   
                    state = "get shape servo plan"

               


        if state == "reset":   
            # rospy.loginfo("**Current state: " + state)
            rospy.logwarn("==== RESETTING ====")
            frame_count = 0
            sample_count = 0
            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], init_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_POS)
            dof_props_2["stiffness"][:8].fill(200.0)
            dof_props_2["damping"][:8].fill(40.0)
            gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)  
            gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, [0,0,0,0,0.15,0,0,0,1.5,0.8]) 

            print("Sucessfully reset robot and object")                
            
            state = "home"
            if fail_mtp:
                state = "home"  
                fail_mtp = False
        
        if sample_count == max_sample_count:  
            frame_count = 0
            sample_count = 0            
            group_count += 1
            print("group count: ", group_count)
            state = "reset" 


        if  goal_point_count >= max_goal_count:                     
            all_done = True 
            final_data = goal_datas
            with open(os.path.join(goal_recording_path, args.obj_name + ".pickle"), 'wb') as handle:
                pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)
            # gym.sync_frame_time(sim)


  
   



    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("total data pt count: ", goal_point_count)
