#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys
sys.path.append('/home/baothach/dvrk_shape_servo/src/dvrk_env/dvrk_gazebo_control/src')

import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from copy import copy
import rospy
# from dvrk_gazebo_control.srv import *
from geometry_msgs.msg import PoseStamped, Pose
from GraspDataCollectionClient import GraspDataCollectionClient
import open3d
# from utils import open3d_ros_helper as orh
# from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
# #import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from utils.isaac_utils import fix_object_frame
import pickle
# from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
# sys.path.append('/home/baothach/shape_servo_DNN')
# # from pointcloud_recon_2 import PointNetShapeServo, PointNetShapeServo2
# from pointcloud_recon_2 import PointNetShapeServo3
# import torch
from robotRRT import RobotRRT
from PIL import Image





ROBOT_Z_OFFSET = 0.20
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 0.86



def get_current_joint_states(i):
    current_position = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)['pos']
    # current_position = [x[0] for x in current_position]
    return list(current_position)

def init():
    for i in range(num_envs):
        # Kuka 1
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][4] = 0.05
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        gym.set_actor_dof_states(envs[i], kuka_handles[i], davinci_dof_states, gymapi.STATE_POS)

        # # Kuka 2
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][4] = 0.05
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
    return list(point_cloud)

def check_reach_desired_position(i, desired_position, error = 0.01 ):
    '''
    Check if the robot has reached the desired goal positions
    '''
    current_position = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)['pos']
    return np.allclose(current_position, desired_position, rtol=0, atol=error)

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
    # seg_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_SEGMENTATION)

    # Get the camera view matrix and invert it to transform points from camera to world
    # space
    
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs_obj[i], cam_handles[0])))

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    # Ignore any points which originate from ground plane or empty space
    # depth_buffer[seg_buffer == 1] = -10001

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
            if p2[0, 2] > 0.005:
                points.append([p2[0, 0], p2[0, 1], p2[0, 2]])

    return np.array(points).astype('float32')

def get_goal_projected_on_image(goal_pc, i, thickness = 0):
    # proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
    # fu = 2/proj[0, 0]
    # fv = 2/proj[1, 1]
    

    u_s =[]
    v_s = []
    for point in goal_pc:
        point = list(point) + [1]

        point = np.expand_dims(np.array(point), axis=0)

        point_cam_frame = point * np.matrix(gym.get_camera_view_matrix(sim, envs_obj[i], vis_cam_handles[0]))
        # print("point_cam_frame:", point_cam_frame)
        # image_coordinates = (gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[0]) * point_cam_frame)
        # print("image_coordinates:",image_coordinates)
        # u_s.append(image_coordinates[1, 0]/image_coordinates[2, 0]*2)
        # v_s.append(image_coordinates[0, 0]/image_coordinates[2, 0]*2)
        # print("fu fv:", fu, fv)
        u_s.append(1/2 * point_cam_frame[0, 0]/point_cam_frame[0, 2])
        v_s.append(1/2 * point_cam_frame[0, 1]/point_cam_frame[0, 2])      
          
    centerU = vis_cam_width/2
    centerV = vis_cam_height/2    
    # print(centerU - np.array(u_s)*cam_width)
    # y_s = (np.array(u_s)*cam_width).astype(int)
    # x_s = (np.array(v_s)*cam_height).astype(int)
    y_s = (centerU - np.array(u_s)*vis_cam_width).astype(int)
    x_s = (centerV + np.array(v_s)*vis_cam_height).astype(int)    

    if thickness != 0:
        new_y_s = deepcopy(list(y_s))
        new_x_s = deepcopy(list(x_s))
        for y, x in zip(y_s, x_s):
            for t in range(1, thickness+1):
                new_y_s.append(max(y-t,0))
                new_x_s.append(max(x-t,0))
                new_y_s.append(max(y-t,0))
                new_x_s.append(min(x+t, vis_cam_height-1))                
                new_y_s.append(min(y+t, vis_cam_width-1))
                new_x_s.append(max(x-t,0))                    
                new_y_s.append(min(y+t, vis_cam_width-1))                
                new_x_s.append(min(x+t, vis_cam_height-1))
        y_s = new_y_s
        x_s = new_x_s
    # print(x_s)
    return x_s, y_s

def visualize_plane(plane_eq, x_range=0.1, y_range=0.5, z_range=0.2,num_pt = 10000):
    plane = []
    for i in range(num_pt):
        x = np.random.uniform(-x_range, x_range)
        z = np.random.uniform(0.03, z_range)
        y = -(plane_eq[0]*x + plane_eq[2]*z + plane_eq[3])/plane_eq[1]
        if -y_range < y < 0:
            plane.append([x, y, z])     
    return plane  

if __name__ == "__main__":



    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--object_type", "type": int, "default": 0, "help": "Type of bjects to place in the bin: 0 - box, 1 - meat can, 2 - banana, 3 - mug, 4 - brick, 5 - random"},
            {"name": "--headless", "type": bool, "default": False, "help": "headless mode"}])

    num_envs = args.num_envs
    



    # configure sim
    sim_type = gymapi.SIM_FLEX
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
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            quit()

    # load robot assets
    asset_root = "../../assets"

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, -two_robot_offset, ROBOT_Z_OFFSET)
    #pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)


    pose_2 = gymapi.Transform()
    pose_2.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    # pose_2.p = gymapi.Vec3(0.0, 0.85, ROBOT_Z_OFFSET)
    pose_2.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

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
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    if sim_type is gymapi.SIM_FLEX:
        asset_options.max_angular_velocity = 40000.

    print("Loading asset '%s' from '%s'" % (kuka_asset_file, asset_root))
    kuka_asset = gym.load_asset(sim, asset_root, kuka_asset_file, asset_options)



    
    # Load soft objects' assets
    asset_root = "/home/baothach/sim_data/BigBird/BigBird_urdf_new" # Current directory
    # soft_asset_file = "soft_box/soft_box.urdf"
    
    # soft_asset_file = "3m_high_tack_spray_adhesive.urdf"
    soft_asset_file = "cheez_it_white_cheddar.urdf"
    # soft_asset_file = "cholula_chipotle_hot_sauce.urdf"
    # asset_root = '/home/baothach/sim_data/Bao_objects/urdf'
    # soft_asset_file = "long_bar.urdf"


    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.0, 0.50-two_robot_offset, 0.03)
    # soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
    soft_thickness = 0.005    # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True
    # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    # print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
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
    






    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    base_poses = []

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add kuka
        kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", i, 1, segmentationId=11)

        # add kuka2
        kuka_2_handle = gym.create_actor(env, kuka_asset, pose_2, "kuka2", i, 1, segmentationId=11)        
        

        # add soft obj        
        env_obj = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs_obj.append(env_obj)        
        
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)
        object_handles.append(soft_actor)




        kuka_handles.append(kuka_handle)
        kuka_handles_2.append(kuka_2_handle)

    # use position and velocity drive for all dofs; override default stiffness and damping values
    dof_props = gym.get_actor_dof_properties(envs[0], kuka_handles[0])
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"][:8].fill(200.0)
    dof_props["damping"][:8].fill(40.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)

    dof_props_2 = gym.get_actor_dof_properties(envs[0], kuka_handles_2[0])
    dof_props_2["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props_2["stiffness"].fill(200.0)
    dof_props_2["damping"].fill(40.0)
    dof_props_2["stiffness"][8:].fill(1)
    dof_props_2["damping"][8:].fill(2)  
    
    # dof_props_2["driveMode"][4].fill(gymapi.DOF_MODE_VEL)
    # dof_props_2["stiffness"][4].fill(0.0)
    # dof_props_2["damping"][4].fill(200.0)

    # Camera setup
    if not args.headless:
        cam_pos = gymapi.Vec3(-1, 0.5, 1)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
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
    cam_positions.append(gymapi.Vec3(0.05, -0.32, 0.25))
    # cam_positions.append(gymapi.Vec3(0.0001, -0.45, 0.22))
    cam_targets.append(gymapi.Vec3(0.0, 0.50-two_robot_offset, 0.00))
    # cam_positions.append(gymapi.Vec3(-0.5, 1.0, 0.5))
    # cam_targets.append(gymapi.Vec3(0.0, 0.4, 0.0))    

    
    for i, env_obj in enumerate(envs_obj):
        # for c in range(len(cam_positions)):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])


    # Camera for point cloud setup
    vis_cam_positions = []
    vis_cam_targets = []
    vis_cam_handles = []
    vis_cam_width = 400
    vis_cam_height = 400
    vis_cam_props = gymapi.CameraProperties()
    vis_cam_props.width = vis_cam_width
    vis_cam_props.height = vis_cam_height



    # vis_cam_positions.append(gymapi.Vec3(-0.13, -0.45, 0.1))   # 6 -1 1 0 0.38
    vis_cam_positions.append(gymapi.Vec3(-0.11, -0.5, 0.1))

    vis_cam_targets.append(gymapi.Vec3(0.0, 0.45-two_robot_offset, 0.05))
    # cam_positions.append(gymapi.Vec3(-0.5, 1.0, 0.5))
    # cam_targets.append(gymapi.Vec3(0.0, 0.4, 0.0))    

    
    for i, env_obj in enumerate(envs_obj):
        # for c in range(len(cam_positions)):
            vis_cam_handles.append(gym.create_camera_sensor(env_obj, vis_cam_props))
            gym.set_camera_location(vis_cam_handles[i], env_obj, vis_cam_positions[0], vis_cam_targets[0])



    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles[i], dof_props)
        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

        

    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')


  


    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    # main_insertion_handle = gym.find_actor_dof_handle(envs[0], kuka_handles[0], 'psm_main_insertion_joint')
    state = "home"
    # state = "get plan"
    # mode = "positive"
    num_image = 0
    sample_count = 0
    frame_count = 0
    max_sample_count = 1000

    final_point_clouds = []
    final_desired_positions = []
    pc_on_trajectory = []
    poses_on_trajectory = []
    first_time = True
    save_intial_pc = True
    # random_stuff = True
    dc_client = GraspDataCollectionClient()
    # switch = True
    max_run_FEM_time = 5
    vis_frame_count = 0
    num_image = 0
    start_vis_cam = False
    prepare_vis_cam = True
    
    # Load multi object poses:
    with open('/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/record_multi_object_poses/batch2(200).pickle', 'rb') as handle:
        saved_object_states = pickle.load(handle)

    
    #    Get goal pc:
    with open('/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/goal_data_combined/sample 0.pickle', 'rb') as handle:
        data = pickle.load(handle)
        goal_pc = data["partial pcs"][1]
  
    # Set up RRT
    lims = np.array([[-1.605, 1.5994], [-0.93556, 0.94249], [-0.94249, 0.93556], [-0.93556,0.94249],
                    [0.042, 0.24], [-3.14, 3.14], [-1.5708, 1.5708], [-1.5708, 1.5708]])
    constrain_plane = np.array([-1, 1, 0, 0.38])
    robot_rrt = RobotRRT(num_samples=500, constrain_plane=constrain_plane, num_dimensions=8, step_length = 1, lims=lims, goal_pc = goal_pc)    
    save_path = "/home/baothach/shape_servo_data/RRT/1"
    
    start_time = timeit.default_timer()    

    close_viewer = False

    # while (not gym.query_viewer_has_closed(viewer)) and (not all_done):
    while (not close_viewer) and (not all_done): 
       
        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        t = gym.get_sim_time(sim)

        if prepare_vis_cam:
            plane_points = visualize_plane(constrain_plane, num_pt=50000)
            plane_xs, plane_ys = get_goal_projected_on_image(plane_points, i, thickness = 0)
            valid_ind = []
            for t in range(len(plane_xs)):
                if 0 < plane_xs[t] < vis_cam_width and 0 < plane_ys[t] < vis_cam_height:
                    valid_ind.append(t)
            plane_xs = np.array(plane_xs)[valid_ind]
            plane_ys = np.array(plane_ys)[valid_ind]
            


            prepare_vis_cam = False

        if start_vis_cam: 
            if vis_frame_count % 20 == 0:
                gym.render_all_camera_sensors(sim)
                im = gym.get_camera_image(sim, envs_obj[i], vis_cam_handles[0], gymapi.IMAGE_COLOR).reshape((vis_cam_height,vis_cam_width,4))
                # goal_xs, goal_ys = get_goal_projected_on_image(data["full pcs"][1], i, thickness = 1)

                im = Image.fromarray(im)
                
                img_path =  os.path.join(save_path, str(num_image)+"_"+str(gym.get_sim_time(sim))+".png")
                
                im.save(img_path)
                num_image += 1

            vis_frame_count += 1


        if state == "home" :   
            rospy.loginfo("**Current state: " + state + ", current sample count: " + str(sample_count))
            frame_count += 1
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.103)
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.203)            
            if frame_count == 10:                      

                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper1_joint"), 1.5)
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper2_joint"), 1.0) 

                # # state = get_new_obj_pose(saved_object_states, num_recorded_poses=200, num_particles_in_obj=1743)
                # idx = 160 #101
                # object_state = saved_object_states[idx*1743:(idx+1)*1743, :]
                # gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(object_state))                  

                state = "generate preshape"
                frame_count = 0

                current_pc = get_point_cloud()
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                pc_ros_msg = fix_object_frame(pc_ros_msg)       
                # gym.refresh_particle_state_tensor(sim)
                # init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                # init_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))  
                # init_joint_states = deepcopy(gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos'][:8])
                # # print("init_joint_states", init_joint_states)
                # ee_pos = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3]["pose"]["p"])
                
                # robot_rrt.save_init_states(init_object_state, init_robot_state, init_joint_states, ee_pos)
                # state = "run RRT"

            
        if state == "generate preshape":                   
             
            cartesian_goal = None
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
            for idx in range(len(preshape_response.palm_goal_pose_world)):  # Pick only top grasp
                if preshape_response.is_top_grasp[idx] == True:
                    cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[idx].pose) # Need fix
                    # dc_clients[i][j].top_grasp_preshape_idx = idx
            if cartesian_goal == None:
                state = "reset"
                rospy.logerr('NO CARTESIAN GOAL.\n') 
                
            else:
                # cartesian_goal.position.x = 0.047#0.023#0.064#-0.072#-0.081#-0.1134
                # cartesian_goal.position.y = -0.239#-0.252#-0.238#-0.341#-0.341#-0.33
                # cartesian_goal.position.z = 0.032#0.032#0.027#0.023#0.024#0.0295  
                
                # cartesian_goal.position.x = 0.075046243#0.032465#0.075046243#-0.0676719#-0.04733923#-0.1134
                # cartesian_goal.position.y = -0.263525#-0.26031256#-0.263525#-0.3105266#-0.3336753#-0.33
                # cartesian_goal.position.z = 0.027#0.032#0.023#0.024#0.0295 
                print("teo:", cartesian_goal.position.x, cartesian_goal.position.y, cartesian_goal.position.z)

                cartesian_goal.position.x = -cartesian_goal.position.x
                cartesian_goal.position.y = -cartesian_goal.position.y
                cartesian_goal.position.z -= ROBOT_Z_OFFSET
                cartesian_goal.orientation.x = 0
                cartesian_goal.orientation.y = 0.707107
                cartesian_goal.orientation.z = 0.707107
                cartesian_goal.orientation.w = 0                

                # Get plan from MoveIt
                dc_client.plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_goal, current_position=get_current_joint_states(i))

                # Does plan exist?
                if (not dc_client.plan_traj):
                    rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                    state = "reset"
                else:
                    rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                    state = "move to preshape"
                    rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))


        if state == "move to preshape":            
            plan_traj_with_gripper = [plan+[1.5,0.8] for plan in dc_client.plan_traj]
            pos_targets = np.array(plan_traj_with_gripper[dc_client.traj_index], dtype=np.float32)
            gym.set_actor_dof_position_targets(envs[i], kuka_handles_2[i], pos_targets)        
            
            dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)
            
            if np.allclose(dof_states['pos'][:8], pos_targets[:8], rtol=0, atol=0.1) and np.allclose(dof_states['pos'][8:], pos_targets[8:], rtol=0, atol=0.1)  :
                dc_client.traj_index += 1             
            
            if dc_client.traj_index == len(dc_client.plan_traj):
                dc_client.traj_index = 0
                state = "generate preshape 2"   
                rospy.loginfo("Succesfully executed PRESHAPE 1 moveit arm plan. Let's fucking grasp it!!")

        if state == "generate preshape 2":       

            cartesian_goal = None
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg, non_random = True)               
            for idx in range(len(preshape_response.palm_goal_pose_world)):  # Pick only top grasp
                if preshape_response.is_top_grasp[idx] == True:
                    cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[idx].pose) # Need fix
                    # dc_clients[i][j].top_grasp_preshape_idx = idx
            if cartesian_goal == None:
                state = "reset"
                rospy.logerr('NO CARTESIAN GOAL.\n') 
                
            else:
                cartesian_goal.position.x = cartesian_goal.position.x
                cartesian_goal.position.y = cartesian_goal.position.y + two_robot_offset
                cartesian_goal.position.z -= ROBOT_Z_OFFSET
                cartesian_goal.orientation.x = 0
                cartesian_goal.orientation.y = 0.707107
                cartesian_goal.orientation.z = 0.707107
                cartesian_goal.orientation.w = 0                

                # Get plan from MoveIt
                dc_client.plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_goal, current_position=get_current_joint_states(i))

                # Does plan exist?
                if (not dc_client.plan_traj):
                    rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                    state = "reset"
                else:
                    rospy.loginfo('Sucesfully found a 2 PRESHAPE moveit plan to grasp.\n')
                    state = "move to preshape 2"
                    rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))


        if state == "move to preshape 2":            
            plan_traj_with_gripper = [plan+[1.5,0.8] for plan in dc_client.plan_traj]
            pos_targets = np.array(plan_traj_with_gripper[dc_client.traj_index], dtype=np.float32)
            gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets)        
            
            dof_states = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
            
            if np.allclose(dof_states['pos'][:8], pos_targets[:8], rtol=0, atol=0.1) and np.allclose(dof_states['pos'][8:], pos_targets[8:], rtol=0, atol=0.1)  :
                dc_client.traj_index += 1             
            
            if dc_client.traj_index == len(dc_client.plan_traj):
                dc_client.traj_index = 0
                state = "grasp object"   
                rospy.loginfo("Succesfully executed PRESHAPE 2 moveit arm plan. Let's fucking grasp it!!")
                

        
        if state == "grasp object":    
                     
            rospy.loginfo("**Current state: " + state)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), -0.5)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), -1.0) 
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper1_joint"), 0.4)
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper2_joint"), -0.3)            

            dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)
            if dof_states['pos'][8] < 0.4:
                                       
                state = "get shape servo plan"
                    
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), 0.35)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), -0.35)         
        
                # anchor_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])
                # print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] )
                
                gym.refresh_particle_state_tensor(sim)
                init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                init_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))  
                init_joint_states = deepcopy(gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos'][:8])
                # print("init_joint_states", init_joint_states)
                ee_pos = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3]["pose"]["p"])
                
                robot_rrt.save_init_states(init_object_state, init_robot_state, init_joint_states, ee_pos)                
                
                state = "run RRT"
                # state = "get final path"


        if state == "run RRT":
            print("**RUNNING RRT")
            if robot_rrt.run_FEM_is_complete:
                if robot_rrt.invalid_state_occur == False:
                    # robot_rrt.T.nodes[-1].get_ee_pos(deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3]["pose"]["p"]))
                    # if robot_rrt.is_valid(robot_rrt.T.nodes[-1]) == False:
                    #     robot_rrt.T.nodes.pop(-1)
                    #     rospy.logwarn("====Invalid!====")
                    # else:
                    gym.refresh_particle_state_tensor(sim)
                    object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))                           
                    robot_rrt.update_node_object_state(object_state, robot_state)
                    

                    if robot_rrt.is_goal(robot_rrt.T.nodes[-1]):
                    # if robot_rrt.is_goal(robot_rrt.T.nodes[-1], goal_oriented=True, threshold = 0.5):
                        robot_rrt.found_path = True
                        rospy.logwarn("====GOAL!====")
                        state = "get final path"
                
                new_node, parent_node = robot_rrt.build_rrt()
                print("new_node.state:", new_node.state)
                if new_node == None:
                    robot_rrt.invalid_state_occur = True
                else:    
                    # Reset robot and object back to the new state's parent node
                    gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(parent_node.object_simulation_state))
                    gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], parent_node.robot_simulation_state, gymapi.STATE_ALL)                    
                    robot_rrt.run_FEM_is_complete = False
                    
                    robot_rrt.invalid_state_occur = False                    
                    rospy.logwarn("====generated a new RRT sample====")
                    start_FEM_time = timeit.default_timer()
            else:
                if timeit.default_timer() - start_FEM_time <= max_run_FEM_time:
                    pos_targets = np.array(list(new_node.state) + [0.35,-0.35], dtype=np.float32)  
                    gym.set_actor_dof_position_targets(envs[i], kuka_handles_2[i], pos_targets)  
                    dof_states = gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos']
                    if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.01):
                        robot_rrt.run_FEM_is_complete = True
                else:
                    robot_rrt.run_FEM_is_complete = True
                    rospy.logwarn("====Running FEM timeout====")
                robot_rrt.T.nodes[-1].state = deepcopy(gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos'][:8])                


        if robot_rrt.reach_max_samples:
            state = "get final path"

            print("reach max sample")


        if state == "get final path":
            start_vis_cam = True
            path = robot_rrt.get_final_path()
    #         path = [np.array([-0.09159944,  0.13329346,  0.05280741, -0.2649262 ,  0.22047678,
    #    -0.05253264, -0.11827098,  0.10448767], dtype=np.float32), np.array([-0.10929127,  0.11509641,  0.05349011, -0.25512323,  0.21928024,
    #    -0.08940128, -0.13698603,  0.11088128], dtype=np.float32), np.array([-0.12404278,  0.10981082,  0.06523064, -0.266626  ,  0.21729553,
    #    -0.14710349, -0.11716215,  0.08006477], dtype=np.float32), np.array([-0.11675891,  0.11253837,  0.06600923, -0.2694087 ,  0.2140482 ,
    #    -0.20125017, -0.08502976,  0.10520224], dtype=np.float32), np.array([-0.10303234,  0.1034482 ,  0.04525568, -0.26995865,  0.21149713,
    #    -0.21622236, -0.05884402,  0.06134045], dtype=np.float32), np.array([-0.10841637,  0.10265822,  0.00501418, -0.2519618 ,  0.20744687,
    #    -0.25894117, -0.09609608,  0.0521715 ], dtype=np.float32), np.array([-0.12788843,  0.08423121,  0.02031196, -0.22832328,  0.20122236,
    #    -0.2641407 , -0.1077749 ,  0.01247364], dtype=np.float32), np.array([-0.1636967 ,  0.08764243,  0.05062798, -0.18290955,  0.19781375,
    #    -0.28390375, -0.09475323,  0.00039351], dtype=np.float32), np.array([-0.19361097,  0.09633575,  0.05073513, -0.1682925 ,  0.19738758,
    #    -0.31432086, -0.06613853,  0.04885097], dtype=np.float32), np.array([-0.19799078,  0.0862848 ,  0.07166246, -0.1686244 ,  0.19831014,
    #    -0.36971885, -0.02741832,  0.03584506], dtype=np.float32), np.array([-0.17501763,  0.07141357,  0.06776635, -0.18330708,  0.20142055,
    #    -0.3787042 ,  0.00199345,  0.03202526], dtype=np.float32), np.array([-0.15493947,  0.05456933,  0.06771223, -0.17929332,  0.19916159,
    #    -0.4339393 ,  0.01776291, -0.00142667], dtype=np.float32), np.array([-0.16283454,  0.04324618,  0.07372724, -0.17350948,  0.19527143,
    #    -0.49631423,  0.02851999, -0.02491721], dtype=np.float32), np.array([-0.15443715,  0.039722  ,  0.05875075, -0.15107583,  0.19674367,
    #    -0.48277044,  0.05618323,  0.00368947], dtype=np.float32), np.array([-0.13369009,  0.05726591,  0.02019149, -0.11816981,  0.19784164,
    #    -0.46146706,  0.0854497 ,  0.00266051], dtype=np.float32), np.array([-0.14174697,  0.02869006,  0.03535777, -0.09411783,  0.19630134,
    #    -0.46866524,  0.10718533,  0.02053458], dtype=np.float32), np.array([-0.1436499 , -0.02288109,  0.02114503, -0.08930441,  0.19288367,
    #    -0.4743046 ,  0.14916386, -0.00232649], dtype=np.float32), np.array([-0.15238027, -0.03485183,  0.02414164, -0.10458837,  0.19151264,
    #    -0.52224106,  0.17586781,  0.04047504], dtype=np.float32), np.array([-0.12991183, -0.01128128, -0.00424043, -0.1279404 ,  0.19174588,
    #    -0.53645974,  0.21890877,  0.06647525], dtype=np.float32), np.array([-1.00862116e-01, -1.73961539e-02, -3.87773471e-05, -1.11125186e-01,
    #     1.90045357e-01, -5.78820229e-01,  2.38073602e-01,  8.63724574e-02],
    #   dtype=np.float32), np.array([-0.10148264, -0.01146917,  0.04096344, -0.08466579,  0.1779111 ,
    #    -0.6015895 ,  0.23654374,  0.12027447], dtype=np.float32), np.array([-0.11348707, -0.0141135 ,  0.0611715 , -0.09939159,  0.18288362,
    #    -0.6563009 ,  0.19542445,  0.13810849], dtype=np.float32), np.array([-0.14078543,  0.00356461,  0.07337437, -0.11852097,  0.1802491 ,
    #    -0.709304  ,  0.21246544,  0.13291065], dtype=np.float32), np.array([-0.58954374,  0.12338046, -0.26262997,  0.16091051,  0.13953235,
    #    -0.98742077, -0.36325211,  0.56048727])]

            print("plan:", path)
            if path is not None:
                # robot_rrt.traj_index = len(path) - 1
                print("Running final path!!!!!!!!!!!")
                if robot_rrt.reset_to_init:
                    gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(robot_rrt.init_obj_state))
                    gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], robot_rrt.init_robot_state, gymapi.STATE_ALL)                   
                    robot_rrt.reset_to_init = False

                robot_state = path[robot_rrt.traj_index]
                dof_states = gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos']
                plan_traj_with_gripper = list(robot_state)+[0.35,-0.35]
                pos_targets = np.array(plan_traj_with_gripper, dtype=np.float32)
                gym.set_actor_dof_position_targets(envs[0], kuka_handles_2[0], pos_targets)                
                
                if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.01):
                    robot_rrt.traj_index += 1 
                    
                if robot_rrt.traj_index == len(path):    
                    state = "done"                
        if state == "done":
            rospy.logwarn("====done!!====")
            print(timeit.default_timer() - start_time)

        # if state == "reset":   
        #     rospy.loginfo("******************Current state: " + state) 
        #     # gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
        #     # pos_targets = np.array([0.,0.,0.,0.,0.05,0.,0.,0.,1.5,0.8], dtype=np.float32)
        #     # gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets)
        #     # gym.set_actor_dof_position_targets(envs[i], kuka_handles_2[i], pos_targets)
        #     # dof_states_1 = gym.get_actor_dof_states(envs[0], kuka_handles[0], gymapi.STATE_POS)['pos']
        #     # dof_states_2 = gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos']
        #     # if np.allclose(dof_states_1, pos_targets, rtol=0, atol=0.1) and np.allclose(dof_states_2, pos_targets, rtol=0, atol=0.1):
        #         # print("Scuesfully reset robot")
            
        #     if switch:          
        #         state = "get shape servo plan"
        #         switch = False
        #     else:
        #         gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
        #         gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], saved_robot_state, gymapi.STATE_ALL)
        #         switch = True
        #         state = "get shape servo plan"
                

                 

            # print("Scuesfully reset robot and object")


        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)
            # gym.sync_frame_time(sim)


  
    # data = {"point clouds": final_point_clouds, "positions": final_desired_positions, "saved intial state":saved_object_state}
    # with open('/home/baothach/shape_servo_data/multi_grasps/one_grasp', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # data = {"init pc": initial_pc, "final pc": goal_pc_numpy, "change": point_cloud_change}
    # with open('/home/baothach/shape_servo_data/point_cloud_change/change_2', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

