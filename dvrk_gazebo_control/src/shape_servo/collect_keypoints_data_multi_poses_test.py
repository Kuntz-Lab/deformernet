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
from utils import open3d_ros_helper as orh
from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
#import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from utils.isaac_utils import fix_object_frame
from utils.record_data_h5 import RecordGraspData_2
import pickle
from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torch



ROBOT_Z_OFFSET = 0.25
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
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(point_cloud)
    # obb = pcd.get_oriented_bounding_box()

    # points = np.asarray(obb.get_box_points())
    # lines = [
    #     [0, 1],
    #     [0, 2],
    #     [0, 3],
    #     [1, 6],
    #     [1, 7],
    #     [2, 5], 
    #     [2, 7],
    #     [3, 5],
    #     [3, 6],
    #     [4, 5],
    #     [4, 6],
    #     [4, 7],
    # ]
    # colors = [[1, 0, 0] for i in range(len(lines))]
    # line_set = open3d.geometry.LineSet(
    #     points=open3d.utility.Vector3dVector(points),
    #     lines=open3d.utility.Vector2iVector(lines),
    # )
    # line_set.colors = open3d.utility.Vector3dVector(colors)
    # open3d.visualization.draw_geometries([pcd, line_set])      

    return list(point_cloud)

def check_reach_desired_position(i, desired_position, error = 0.01 ):
    '''
    Check if the robot has reached the desired goal positions
    '''
    current_position = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)['pos']
    return np.allclose(current_position, desired_position, rtol=0, atol=error)
    
def get_depth_image_and_point_cloud(i):

    # Render all of the image sensors only when we need their output here
    # rather than every frame.
    gym.render_all_camera_sensors(sim)

    
   
    print("Converting Depth images to point clouds. Have patience...")
    # for c in range(len(cam_handles)):
    
    # print("Deprojecting from camera %d, %d" % i))
    # Retrieve depth and segmentation buffer
    depth_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_SEGMENTATION)
    # print(type(depth_buffer))
    # print(depth_buffer.shape)
    points = np.zeros((depth_buffer.shape[0], depth_buffer.shape[1], 3))

    # Get the camera view matrix and invert it to transform points from camera to world
    # space
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs_obj[i], cam_handles[0])))

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    # Ignore any points which originate from ground plane or empty space
    # depth_buffer[seg_buffer == 1] = -3

    centerU = cam_width/2
    centerV = cam_height/2
    filter_value = -3
    depth_buffer[depth_buffer <= filter_value] = filter_value
    for k in range(cam_width):
        for t in range(cam_height):
            if depth_buffer[t, k] == filter_value:
                # depth_buffer[t, k] = max_depth
                continue
                
            # if seg_buffer[t, k] == 0:
            u = -(k-centerU)/(cam_width)  # image-space coordinate
            v = (t-centerV)/(cam_height)  # image-space coordinate
            d = depth_buffer[t, k]  # depth buffer value
            X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
            p2 = X2*vinv  # Inverse camera view to get world coordinates
            if p2[0, 2] > 0.01:
                # points.append([p2[0, 0], p2[0, 1], p2[0, 2]])
                points[t, k] = [p2[0, 0], p2[0, 1], p2[0, 2]]
            else:
                depth_buffer[t, k] = filter_value
    
   
    return points, convert_depth_map_to_png(-depth_buffer, filter_value)        

def convert_depth_map_to_png(depth_map, filter_value):
    flatten_depth_map = depth_map.flatten()
    flatten_depth_map = [x for x in list(flatten_depth_map) if x != -filter_value]
    max_depth = max(flatten_depth_map)
    # print(max_depth)

    min_depth = min(flatten_depth_map)

    depth_map = np.where(depth_map == -filter_value, min_depth, depth_map)
    png_depth_map = (depth_map - min_depth)*255/(max_depth - min_depth)
    return png_depth_map

def get_new_obj_pose(saved_object_states, num_recorded_poses, num_particles_in_obj):
    choice = np.random.randint(0, num_recorded_poses)   
    state = saved_object_states[choice*num_particles_in_obj : (choice+1)*num_particles_in_obj, :] 
    return state        # torch size (1743, 3)   
           
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
    dc_client = GraspDataCollectionClient()
  
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
        
        # soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)
        # object_handles.append(soft_actor)

        # add soft obj (multi poses)
        object_pose_stamped = dc_client.gen_object_pose()
        object_pose = gymapi.Transform()
        
        delta_x = -0.01 #np.random.uniform(low = -0.03, high = 0.03)
        delta_y = 0.03 #np.random.uniform(low = -0.03, high = 0.03)

        object_pose.p = gymapi.Vec3(0.0 + delta_x, 0.50-two_robot_offset + delta_y, 0.03)
        object_pose.r = gymapi.Quat(object_pose_stamped.pose.orientation.x,object_pose_stamped.pose.orientation.y,\
                                        object_pose_stamped.pose.orientation.z,object_pose_stamped.pose.orientation.w) 
        soft_actor = gym.create_actor(env_obj, soft_asset, object_pose, "soft", i, 0)
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
        cam_pos = gymapi.Vec3(1, 0.5, 1)
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
    cam_positions.append(gymapi.Vec3(0.17, -0.41, 0.25))
    # cam_positions.append(gymapi.Vec3(0.0001, -0.45, 0.22))
    cam_targets.append(gymapi.Vec3(0.0, 0.50-two_robot_offset, 0.00))
    # cam_positions.append(gymapi.Vec3(-0.5, 1.0, 0.5))
    # cam_targets.append(gymapi.Vec3(0.0, 0.4, 0.0))    

    
    for i, env_obj in enumerate(envs_obj):
        # for c in range(len(cam_positions)):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])



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
    main_insertion_handle = gym.find_actor_dof_handle(envs[0], kuka_handles[0], 'psm_main_insertion_joint')
    state = "home"
    
    i = 0
    image_pair_count = 0
    sample_count = 0
    frame_count = 0
    group_count = 0
    max_group_count = 2
    max_sample_count = 2
    max_data_point_count = 10000

    final_point_clouds = []
    final_desired_positions = []
    pc_on_trajectory = []
    poses_on_trajectory = []
    first_time = True
    save_intial_pc = True
    # reset_soft_obj = True
    # random_stuff = True
    image_recording_path = "/home/baothach/shape_servo_data/keypoints/batch7/depth"
    source_image_path = os.path.join(image_recording_path, "source")
    target_image_path = os.path.join(image_recording_path, "target")

    # Load multi object poses:
    with open('/home/baothach/shape_servo_data/keypoints/combined_w_shape_servo/record_multi_object_poses/batch2(200).pickle', 'rb') as handle:
        saved_object_states = pickle.load(handle)    


    # initilize h5 file
    record_grasp_data = RecordGraspData_2()
    
    
    start_time = timeit.default_timer()    

    close_viewer = False

    # while (not gym.query_viewer_has_closed(viewer)) and (not2 all_done):
    while (not close_viewer) and (not all_done): 
        
        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        t = gym.get_sim_time(sim)
 
        if state == "home" :   
            rospy.loginfo("**Current state: " + state + ", current sample count: " + str(sample_count))
      
            frame_count += 1
            if frame_count <= 6:                             
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.103)
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.203)
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper1_joint"), 1.5)
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper2_joint"), 1.0) 

            else:
            # state = get_new_obj_pose(saved_object_states, num_recorded_poses=200, num_particles_in_obj=1743)
            # gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(state))                  

                state = "generate preshape"
                frame_count = 0

                current_pc = get_point_cloud()
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                pc_ros_msg = fix_object_frame(pc_ros_msg)
            
        


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
                rospy.loginfo("Succesfully executed PRESHAPE moveit arm plan. Let's fucking grasp it!!")

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
                    rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
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
                rospy.loginfo("Succesfully executed PRESHAPE moveit arm plan. Let's fucking grasp it!!")



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
        
                current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])
                print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] )



        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state)

 
            # gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_tool_gripper1_joint"), 0.4)
            # gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_tool_gripper2_joint"), -0.3) 

            # Interpolation: https://en.wikipedia.org/wiki/Linear_interpolation
            # max_x = 0.05 + (current_pose["pose"]["p"]["y"]+0.43)*(0.09-0.05)/(-0.375+0.43)
            # max_y = 0.03 + (current_pose["pose"]["p"]["y"]+0.43)*(0.06-0.03)/(-0.375+0.43)              
            # max_z = 0.06 + (current_pose["pose"]["p"]["y"]+0.43)*(0.09-0.06)/(-0.375+0.43)
            # max_x = 0.06 
            # max_y = 0.05               
            # max_z = 0.07    

            delta_x = -0.00   
            delta_y = 0.03 
            delta_z = 0.08  
            # print("max x, y, z:", max_x, max_y, max_z)
            # delta_x = np.random.uniform(low = -max_x, high = max_x)   
            # delta_y = np.random.uniform(low = 0.0, high = max_y)
            # delta_z = np.random.uniform(low = 0.0, high = max_z) 
           

            cartesian_pose = Pose()
            cartesian_pose.orientation.x = 0
            cartesian_pose.orientation.y = 0.707107
            cartesian_pose.orientation.z = 0.707107
            cartesian_pose.orientation.w = 0
            cartesian_pose.position.x = -current_pose["pose"]["p"]["x"] + delta_x
            cartesian_pose.position.y = -current_pose["pose"]["p"]["y"] + delta_y
            cartesian_pose.position.z = current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET + delta_z
            dof_states = gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos']

            plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=dof_states)
            if (not plan_traj):
                rospy.logerr('Can not find moveit plan to shape servo. Ignore this grasp.\n')  
                # state = "reset"
            else:
                state = "move to goal"
                traj_index = 0

        if state == "move to goal":
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if (not(20 in contacts or 21 in contacts) or not(9 in contacts and 10 in contacts)):  # lose contact w robot 2 or robot 1
                print("Lost contact with robot")
                         
                
                # image_pair_count += 1
                group_count += 1
                state = "record data"

            else:
                # if frame_count % 10 == 0:
                if frame_count == 0:                        
                    source_pc, source_depth_image = get_depth_image_and_point_cloud(i)
                    source_depth_image = Image.fromarray(np.uint8(source_depth_image), mode = 'L')
        
                frame_count += 1           
                
                
                dof_states = gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos']
                plan_traj_with_gripper = [plan+[0.35,-0.35] for plan in plan_traj]
                pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
                gym.set_actor_dof_position_targets(envs[0], kuka_handles_2[0], pos_targets)                
                

                if traj_index <= len(plan_traj) - 2:
                    if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.1):
                        traj_index += 1 
                else:
                    if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.01):
                        traj_index += 1   

                if traj_index == len(plan_traj):
                    traj_index = 0  
                    rospy.loginfo("Succesfully executed moveit arm plan. Let's get to the next shape servo plan!!")   
                             
                    pc, depth_image = get_depth_image_and_point_cloud(i)
                    depth_image = Image.fromarray(np.uint8(depth_image), mode = 'L')  
                    
                    record_grasp_data.handle_record_grasp_data(source_pc, 'source') 
                    record_grasp_data.handle_record_grasp_data(pc, 'target')    

                    source_depth_image.save(os.path.join(source_image_path, "image_" + str(image_pair_count) + ".png"))
                    depth_image.save(os.path.join(target_image_path, "image_" + str(image_pair_count) + ".png"))
                                   

                    
                    frame_count = 0
                    image_pair_count += 1
                    sample_count += 1
                    print("group ", group_count, ", sample ", sample_count)
                    state = "get shape servo plan"
          



        if state == "reset":   
            rospy.loginfo("**Current state: " + state)
            frame_count = 0

            pos_targets = np.array([0.,0.,0.,0.,0.05,0.,0.,0.,1.5,0.8], dtype=np.float32)
            gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets)
            gym.set_actor_dof_position_targets(envs[i], kuka_handles_2[i], pos_targets)
            dof_states_1 = gym.get_actor_dof_states(envs[0], kuka_handles[0], gymapi.STATE_POS)['pos']
            dof_states_2 = gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos']
            # if np.allclose(dof_states_1[:8], pos_targets[:8], rtol=0, atol=0.03) and np.allclose(dof_states_1[8:], pos_targets[8:], rtol=0, atol=0.5):
            if np.allclose(dof_states_1, pos_targets, rtol=0, atol=0.1) and np.allclose(dof_states_2, pos_targets, rtol=0, atol=0.1):
                
                # state = get_new_obj_pose(saved_object_states, num_recorded_poses=100, num_particles_in_obj=1743)
                # gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(state))  
                print("Succesfully reset robot and object")
                # pc_on_trajectory = []
                # poses_on_trajectory = []
                # final_point_clouds = []
                # final_desired_positions = []
                
                state = "home"
 
  
   
                
                
                
        # if sample_count % 5 == 0:
        #     data = {"point clouds": final_point_clouds, "positions": final_desired_positions, "saved intial state":saved_object_state}
        #     with open('/home/baothach/shape_servo_data/multi_grasps/one_grasp', 'wb') as handle:
        #         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)            

        
        if sample_count == max_sample_count:  
            sample_count = 0            
            group_count += 1
            print("group count: ", group_count)
            state = "record data" 

        if state == "record data":
            frame_count = 0
            sample_count = 0
            # record_grasp_data.handle_record_grasp_data(grasp_pose = current_pose["pose"], point_clouds = final_point_clouds, positions = final_desired_positions)
            rospy.loginfo("succesfully record data")  
            state = "reset"

        if group_count == max_group_count or image_pair_count >= max_data_point_count: 
            all_done = True 

        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)
            # gym.sync_frame_time(sim)


  
    # data = {"point clouds": final_point_clouds, "positions": final_desired_positions, "saved intial state":saved_object_state}
    # with open('/home/baothach/shape_servo_data/multi_grasps/goal_1a', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



    print("All done !")
    print("total image pairs: ", image_pair_count)
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

