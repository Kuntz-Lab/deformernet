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
from dvrk_gazebo_control.srv import *
from geometry_msgs.msg import PoseStamped, Pose
import open3d
from utils import open3d_ros_helper as orh
from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
#import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from dnn_architecture import Net, train
import torch
import torch.optim as optim
from ShapeServo import *
from sklearn.decomposition import PCA
import timeit

ROBOT_Z_OFFSET = 0.1
angle_kuka_2 = -0.4
init_kuka_2 = 0.15


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

def init():
    for i in range(num_envs):
        # Kuka 1
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][4] = 0.15
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 1
        gym.set_actor_dof_states(envs[i], kuka_handles[i], davinci_dof_states, gymapi.STATE_POS)

        # Kuka 2
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_NONE)
        # for j in range(8):
        #     davinci_dof_states['pos'][j] = 0.0
        davinci_dof_states['pos'][3] = angle_kuka_2
        davinci_dof_states['pos'][4] = init_kuka_2
        davinci_dof_states['pos'][8] = 0.4
        davinci_dof_states['pos'][9] = -0.3
        gym.set_actor_dof_states(envs[i], kuka_handles_2[i], davinci_dof_states, gymapi.STATE_POS)

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
    dof_props["stiffness"].fill(1000.0)
    dof_props["damping"].fill(200.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)

    dof_props_2 = gym.get_actor_dof_properties(envs[0], kuka_handles_2[0])
    dof_props_2["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props_2["stiffness"].fill(1000.0)
    dof_props_2["damping"].fill(200.0)
    dof_props_2["stiffness"][8:].fill(1)
    dof_props_2["damping"][8:].fill(2)  
    
    # dof_props_2["driveMode"][4].fill(gymapi.DOF_MODE_VEL)
    # dof_props_2["stiffness"][4].fill(0.0)
    # dof_props_2["damping"][4].fill(200.0)

    # Camera setup
    cam_pos = gymapi.Vec3(1, 0.5, 1)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
    middle_env = envs[num_envs // 2 + num_per_row // 2]
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



    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles[i], dof_props)
        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

        

    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')

    # Set up PCA:
    with open("/home/baothach/shape_servo_data/extended_FPFH_vector_135.txt", 'rb') as f:
        FPFH_135_vectors = np.array(pickle.load(f))    
    pca = PCA(n_components=30)
    pca.fit(FPFH_135_vectors)

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    main_insertion_handle = gym.find_actor_dof_handle(envs[0], kuka_handles_2[0], 'psm_main_insertion_joint')
    state = "home"
    with open("/home/baothach/shape_servo_data/goal_feature_vector_30.txt", 'rb') as f:
        goal = np.array(pickle.load(f))  
    with open("/home/baothach/shape_servo_data/goal_points.txt", 'rb') as f:
        goal_points = np.array(pickle.load(f))  
    with open("/home/baothach/shape_servo_data/goal_points_initial.txt", 'rb') as f:
        goal_points_initial = np.array(pickle.load(f))
    with open("/home/baothach/shape_servo_data/goal_initial_135.txt", 'rb') as f:
        goal_initial_135 = np.array(pickle.load(f))
    with open("/home/baothach/shape_servo_data/goal_initial_30.txt", 'rb') as f:
        goal_initial_30 = np.array(pickle.load(f))



    first_time_step = True 
    frame_count = 0
    
    start_time = timeit.default_timer()
    while (not gym.query_viewer_has_closed(viewer)) and (not all_done):

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        t = gym.get_sim_time(sim)
 
        if state == "home" :   
            rospy.loginfo("**Current state: " + state)
            if t <= 1:
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.213)
                # gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper1_joint"), 1.5)
                # gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper2_joint"), 1.0)                

                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_pitch_end_joint"), angle_kuka_2)
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), init_kuka_2)
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_tool_gripper1_joint"), 0.4)
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_tool_gripper2_joint"), -0.3) 
            
            elif t <= 2:
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper1_joint"), 0.7)
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper2_joint"), 0.0) 
                
            else:
                before_servo_pos = 0.225
                gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), before_servo_pos)
                main_ins_pos = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"))
                
                dof_states = gym.get_actor_dof_states(envs[0], kuka_handles[0], gymapi.STATE_POS)
                
                if np.allclose(main_ins_pos, before_servo_pos, atol=0.002) and np.allclose(dof_states['pos'][8:], [0.8, 0.22], rtol=0, atol=0.005):
                    gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper1_joint"), 0.8)
                    gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_tool_gripper2_joint"), 0.22)                     
                    # state = "move to goal"
                    state = "get point cloud initial"

        if state == "get point cloud initial":
            rospy.loginfo("**Current state: " + state)

            # Array of RGB Colors, one per camera, for dots in the resulting
            # point cloud. Points will have a color which indicates which camera's
            # depth image created the point.
            color_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]])

            # Render all of the image sensors only when we need their output here
            # rather than every frame.
            gym.render_all_camera_sensors(sim)

            points_inital = []
            color = []
            print("Converting Depth images to point clouds. Have patience...")
            # for c in range(len(cam_handles)):
            
            print("Deprojecting from camera %d" % i)
            # Retrieve depth and segmentation buffer
            depth_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_DEPTH)
            seg_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_SEGMENTATION)

            # Get the camera view matrix and invert it to transform points from camera to world
            # space
            vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs_obj[0], cam_handles[0])))

            # Get the camera projection matrix and get the necessary scaling
            # coefficients for deprojection
            proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
            fu = 2/proj[0, 0]
            fv = 2/proj[1, 1]

            # Ignore any points which originate from ground plane or empty space
            depth_buffer[seg_buffer == 1] = -10001

            centerU = cam_width/2
            centerV = cam_height/2
            for k in range(cam_width):
                for j in range(cam_height):
                    if depth_buffer[j, k] < -1:
                        continue
                    if seg_buffer[j, k] == 0:
                        u = -(k-centerU)/(cam_width)  # image-space coordinate
                        v = (j-centerV)/(cam_height)  # image-space coordinate
                        d = depth_buffer[j, k]  # depth buffer value
                        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                        p2 = X2*vinv  # Inverse camera view to get world coordinates
                        if p2[0, 2] > 0.005:
                            points_inital.append([p2[0, 0], p2[0, 1], p2[0, 2]])
                            color.append(0)
            with open("/home/baothach/shape_servo_data/goal_points_initial.txt", 'wb') as f:
                pickle.dump(points_inital, f)  
            state = "move to goal"

        if state == "move to goal":
  

            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.24)
 
            main_ins_pos = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"))            
            if np.allclose(main_ins_pos, 0.24, atol=0.003):                  
                state = "get point cloud"

        if state == "get point cloud":
            rospy.loginfo("**Current state: " + state)

            # Array of RGB Colors, one per camera, for dots in the resulting
            # point cloud. Points will have a color which indicates which camera's
            # depth image created the point.
            color_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]])

            # Render all of the image sensors only when we need their output here
            # rather than every frame.
            gym.render_all_camera_sensors(sim)

            points = []
            color = []
            print("Converting Depth images to point clouds. Have patience...")
            # for c in range(len(cam_handles)):
            
            print("Deprojecting from camera %d" % i)
            # Retrieve depth and segmentation buffer
            depth_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_DEPTH)
            seg_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_SEGMENTATION)

            # Get the camera view matrix and invert it to transform points from camera to world
            # space
            vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs_obj[0], cam_handles[0])))

            # Get the camera projection matrix and get the necessary scaling
            # coefficients for deprojection
            proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
            fu = 2/proj[0, 0]
            fv = 2/proj[1, 1]

            # Ignore any points which originate from ground plane or empty space
            depth_buffer[seg_buffer == 1] = -10001

            centerU = cam_width/2
            centerV = cam_height/2
            for k in range(cam_width):
                for j in range(cam_height):
                    if depth_buffer[j, k] < -1:
                        continue
                    if seg_buffer[j, k] == 0:
                        u = -(k-centerU)/(cam_width)  # image-space coordinate
                        v = (j-centerV)/(cam_height)  # image-space coordinate
                        d = depth_buffer[j, k]  # depth buffer value
                        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                        p2 = X2*vinv  # Inverse camera view to get world coordinates
                        if p2[0, 2] > 0.005:
                            points.append([p2[0, 0], p2[0, 1], p2[0, 2]])
                            color.append(0)
            
            # pcd_vis = open3d.geometry.PointCloud()
            # pcd_vis.points = open3d.utility.Vector3dVector(np.array(points))
            # open3d.io.write_point_cloud("/home/baothach/shape_servo_data/visualization/goal_point_cloud.pcd", pcd_vis)
            # open3d.visualization.draw_geometries([pcd_vis])                
            
            state = "get feature vector from VFH and PCA"

        if state == "get feature vector from VFH and PCA":
            rospy.loginfo("**Current state: " + state)
            # Get feature vector
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np.array(points))            
            ros_cloud = orh.o3dpc_to_rospc(pcd)
            feature_vector_135 = np.array(VFH_client(ros_cloud)).reshape(1, -1)  # np array size (1,135)
            feature_vector_30 = pca.transform(feature_vector_135) # np array size (1,30)

            # Get feature vector for initial position
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np.array(points_inital))            
            ros_cloud = orh.o3dpc_to_rospc(pcd)
            feature_vector_135_initial = np.array(VFH_client(ros_cloud)).reshape(1, -1)  # np array size (1,135)
            feature_vector_30_initial = pca.transform(feature_vector_135_initial) # np array size (1,30)            
            
            distance1 = np.linalg.norm(goal - feature_vector_30)
            distance2 = np.linalg.norm(np.array(points.pop()) - np.array(goal_points))
            distance3 = np.linalg.norm(np.array(points_inital.pop()) - np.array(goal_points_initial))
            distance4 = np.linalg.norm(goal_initial_30 - feature_vector_30_initial)
            distance5 = np.linalg.norm(goal_initial_135 - feature_vector_135_initial)
            distance_init_to_final = np.linalg.norm(goal - goal_initial_30)
            distance_init_to_final_goal = np.linalg.norm(feature_vector_30 - feature_vector_30_initial)
            
            print("***distance: ", distance1) 
            print("***distance: ", distance2) 
            print("***distance initial (point): ", distance3) 
            print("***distance initial (30): ", distance4)
            print("***distance initial (135): ", distance5)
            print("init to final (current): ", distance_init_to_final)
            print("init to final (goal): ", distance_init_to_final_goal)

          
                    
            with open("/home/baothach/shape_servo_data/goal_feature_vector_30.txt", 'wb') as f:
                pickle.dump(list(feature_vector_30), f)              

            with open("/home/baothach/shape_servo_data/goal_feature_vector_135.txt", 'wb') as f:
                pickle.dump(list(feature_vector_135), f)  

            with open("/home/baothach/shape_servo_data/goal_points.txt", 'wb') as f:
                pickle.dump(points, f)     

            with open("/home/baothach/shape_servo_data/goal_initial_30.txt", 'wb') as f:
                pickle.dump(list(feature_vector_30_initial), f)      

            with open("/home/baothach/shape_servo_data/goal_initial_135.txt", 'wb') as f:
                pickle.dump(list(feature_vector_135_initial), f)            

            state = "done"



            
        # step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

   
    visualization_data = {"points": final_points, "135": final_vtc_135, "30": final_vtc_30}
    with open('/home/baothach/shape_servo_data/visualization/shape_servo_vis', 'wb') as handle:
        pickle.dump(visualization_data, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

