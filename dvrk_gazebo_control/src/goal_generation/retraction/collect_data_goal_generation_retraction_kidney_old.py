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

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl, TaskVelocityControl2
import transformations
import torch
from sklearn.neighbors import NearestNeighbors


ROBOT_Z_OFFSET = 0.25

two_robot_offset = 0.86
from goal_plane import *

# sys.path.append('/home/baothach/shape_servo_DNN')
# from farthest_point_sampling import *

def init():
    for i in range(num_envs):
        # # Kuka 2
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        davinci_dof_states['pos'][4] = 0.22
        gym.set_actor_dof_states(envs[i], kuka_handles_2[i], davinci_dof_states, gymapi.STATE_POS)

def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

def get_point_cloud():
    gym.refresh_particle_state_tensor(sim)
    particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
    point_cloud = particle_state_tensor.numpy()[:, :3]  
    
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(point_cloud))
    # open3d.visualization.draw_geometries([pcd])     
    # return list(point_cloud)
    return point_cloud.astype('float32')


def get_partial_point_cloud(obj_name="tissue"):

    if obj_name == "tissue":
        cam_handles = tissue_cam_handles
        cam_width = tissue_cam_width
        cam_height = tissue_cam_height
    elif obj_name == "kidney":
        cam_handles = kidney_cam_handles
        cam_width = kidney_cam_width
        cam_height = kidney_cam_height

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
    if obj_name == "tissue":
        depth_buffer[seg_buffer == 11] = -10001
        depth_buffer[seg_buffer == 10] = -10001
    elif obj_name == "kidney":   
        depth_buffer[seg_buffer != 10] = -10001 

    centerU = cam_width/2
    centerV = cam_height/2
    for k in range(cam_width):
        for t in range(cam_height):
            if obj_name == "tissue":
                if depth_buffer[t, k] < -3:
                    continue
            elif obj_name == "kidney":
                if depth_buffer[t, k] < -0.3:
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
            {"name": "--obj_name", "type": str, "default": 'box_0', "help": "select variations of a primitive shape"},
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"},
            {"name": "--kidney_idx", "type": int, "default": 0, "help": "which kidney to use?"},])
    
    num_envs = args.num_envs
    args.headless = args.headless == "True"
    


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
        
        # if args.data_category == "deformernet":
        #     sim_params.flex.num_outer_iterations = 10
        # elif args.data_category == "MP":
        #     sim_params.flex.num_outer_iterations = 4

        sim_params.flex.num_outer_iterations = 6 #4
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

    rigid_asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/goal_generation/muti_kidneys"
    # rigid_asset_file = "kidney_rigid.urdf"
    rigid_asset_file = f"kidney_{args.kidney_idx}.urdf"
    rigid_pose = gymapi.Transform()
    rigid_pose.p = gymapi.Vec3(0.00, 0.40-two_robot_offset, 0.03)
    # rigid_pose.r = gymapi.Quat(0.0, 0.0, -0.707107, 0.707107)
    
    kidney_angle, tissue_angle = get_kidney_and_tissue_angle()
    # kidney_angle = -np.pi/4.5
    eulers = [-np.pi/2+kidney_angle, 0, np.pi]
    quat = transformations.quaternion_from_euler(*eulers)
    rigid_pose.r = gymapi.Quat(*list(quat))
    
    asset_options.thickness = 0.003 # 0.002
    rigid_asset = gym.load_asset(sim, rigid_asset_root, rigid_asset_file, asset_options)



    
    # Load soft objects' assets

    asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/goal_generation/"
    soft_asset_file = 'thin_tissue_layer_attached.urdf'


    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.035, 0.37-two_robot_offset, 0.1) 
    
    # tissue_angle = -np.pi/4.5
    eulers = [np.pi/2+tissue_angle, 0, np.pi]
    quat = transformations.quaternion_from_euler(*eulers)
    soft_pose.r = gymapi.Quat(*list(quat))
      
    soft_thickness = 0.001    # important to add some thickness to the soft body to avoid interpenetrations



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
    

    print("Creating %d environments" % num_envs)
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
        envs_obj.append(env_obj)        
        
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)

        # add rigid obj
        rigid_actor = gym.create_actor(env, rigid_asset, rigid_pose, 'rigid', i, 0, segmentationId=10)
        color = gymapi.Vec3(1,0,0)
        gym.set_rigid_body_color(env, rigid_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object_handles.append(soft_actor)


        kuka_handles_2.append(kuka_2_handle)



    dof_props_2 = gym.get_asset_dof_properties(kuka_asset)
    dof_props_2["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props_2["stiffness"].fill(200.0)
    dof_props_2["damping"].fill(40.0)
    dof_props_2["stiffness"][8:].fill(1)
    dof_props_2["damping"][8:].fill(2)  
    vel_limits = dof_props_2['velocity']  
    # print("======vel_limits:", vel_limits)  
    

    # Camera setup
    if not args.headless:
        cam_pos = gymapi.Vec3(1, 0.5, 1)
        cam_target = gymapi.Vec3(0.0, -0.36, 0.1)

        cam_pos = gymapi.Vec3(0.15, 0.17-two_robot_offset, 0.2)
        cam_target = gymapi.Vec3(0.035, 0.37-two_robot_offset, 0.05)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    # tissue_camera for tissue point cloud
    tissue_cam_positions = []
    tissue_cam_targets = []
    tissue_cam_handles = []
    tissue_cam_width = 256
    tissue_cam_height = 256
    tissue_cam_props = gymapi.CameraProperties()
    tissue_cam_props.width = tissue_cam_width
    tissue_cam_props.height = tissue_cam_height

    # tissue_cam_positions.append(gymapi.Vec3(0.17, -0.62, 0.2))
    # tissue_cam_targets.append(gymapi.Vec3(0.0, 0.40-two_robot_offset, 0.01))
   
    
    tissue_cam_positions.append(gymapi.Vec3(0.15, 0.17-two_robot_offset, 0.2))
    tissue_cam_targets.append(gymapi.Vec3(0.035, 0.37-two_robot_offset, 0.05))
    
    # tissue_cam_positions.append(gymapi.Vec3(0.15, 0.22-two_robot_offset, 0.25))
    # tissue_cam_targets.append(gymapi.Vec3(0.035, 0.37-two_robot_offset, 0.05))

    
    for i, env_obj in enumerate(envs_obj):
            tissue_cam_handles.append(gym.create_camera_sensor(env_obj, tissue_cam_props))
            gym.set_camera_location(tissue_cam_handles[i], env_obj, tissue_cam_positions[0], tissue_cam_targets[0])


    # kidney_camera for tissue point cloud
    kidney_cam_positions = []
    kidney_cam_targets = []
    kidney_cam_handles = []
    kidney_cam_width = 300
    kidney_cam_height = 300    
    kidney_cam_props = gymapi.CameraProperties()
    kidney_cam_props.width = kidney_cam_width
    kidney_cam_props.height = kidney_cam_height

    kidney_cam_positions.append(gymapi.Vec3(-0.12, 0.60-two_robot_offset, 0.1))
    kidney_cam_targets.append(gymapi.Vec3(0.00, 0.40-two_robot_offset, 0.03))
    
        
    for i, env_obj in enumerate(envs_obj):
            kidney_cam_handles.append(gym.create_camera_sensor(env_obj, kidney_cam_props))
            gym.set_camera_location(kidney_cam_handles[i], env_obj, kidney_cam_positions[0], kidney_cam_targets[0])


    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

        

    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')
    rospy.logerr(f"======Loading object ... kidkey_{str(args.kidney_idx)}")  
 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    

    data_recording_path = "/home/baothach/shape_servo_data/goal_generation/retraction_kidney/data"
    os.makedirs(data_recording_path, exist_ok=True)


    terminate_count = 0
    
    frame_count = 0
    data_point_count = len(os.listdir(data_recording_path))
    max_data_point_count = 10000

    rospy.logwarn(f"data_point_count: {data_point_count}")

    


    final_point_clouds = []
    final_desired_positions = []
    pc_on_trajectory = []
    full_pc_on_trajectory = [] 
    # curr_trans_on_trajectory = []
    


    dc_client = GraspDataCollectionClient()
    
    
    ### Plane stuff
    num_new_goal = 0
    max_num_new_goal = 3    #3    
    sample_count = 0
    max_sample_count = 3
    max_plane_time = 2 * 60 # 2 mins
    first_action = True
    first_time = True
    visualization = False
    

    
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
            # gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.103)
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.22)            
            if frame_count == 5:
                rospy.loginfo("**Current state: " + state + ", current sample count: " + str(sample_count))
                

                if first_time:                    
                    frame_state = gym.get_actor_rigid_body_states(envs[i], object_handles[i], gymapi.STATE_POS)                
                    frame_state['pose']['p']['z'] -= 0.05                
                    gym.set_actor_rigid_body_states(envs[i], object_handles[i], frame_state, gymapi.STATE_ALL) 
                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                    
                    # first_time = False

            if frame_count == 7:
                if first_time:
                    current_pc = get_point_cloud()
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                    open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                    pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                    pc_ros_msg = fix_object_frame(pc_ros_msg)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
                    # open3d.visualization.draw_geometries([pcd]) 
                

            if frame_count == 10:
                state = "generate preshape"
                
                frame_count = 0
                
                if first_time:
                    # constrain_plane = np.array([-np.tan(kidney_angle), 1, 0, -(0.40-0.86)+0.02])  
                    constrain_plane = np.array([-np.tan(kidney_angle), 1, 0, -(0.40-0.86)])               
                    kidney_pc = get_partial_point_cloud(obj_name="kidney")
                    first_time = False
                
                if visualization:
                    pcd = pcd_ize(get_partial_point_cloud(obj_name="tissue"))
                    pcd.paint_uniform_color([0,0,0])
                    pcd_plane = pcd_ize(visualize_plane(constrain_plane, x_range=0.15, y_range=[-0.57,-0.27], z_range=[0.0,0.18], num_pts=2000))
                    pcd_plane.paint_uniform_color([1,0,0])
                    pcd_kidney = pcd_ize(kidney_pc)
                    pcd_kidney.paint_uniform_color([0,1,0])
                    open3d.visualization.draw_geometries([pcd, pcd_plane, pcd_kidney])

        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
            cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            target_pose = [-cartesian_goal.position.x, -cartesian_goal.position.y, cartesian_goal.position.z-ROBOT_Z_OFFSET,
                            0, 0.707107, 0.707107, 0]

            mtp_behavior = MoveToPose(target_pose, robot, sim_params.dt, 2)
            if mtp_behavior.is_complete_failure():
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "reset"                
            else:
                rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                state = "move to preshape"
                # rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))


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

                _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                init_eulers = transformations.euler_from_matrix(init_pose)

                dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                dof_props_2["stiffness"][:8].fill(0.0)
                dof_props_2["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)
                
                plane_start_time = timeit.default_timer()



        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 

            if first_action:
                delta_x, delta_y, delta_z = get_action(tissue_angle, y_mag=0.12, z_mag=0.08)
                first_action = False
            else:
                delta_x, delta_y, delta_z = get_action(tissue_angle, y_mag=0.03, z_mag=0.0)


            tvc_behavior = TaskVelocityControl([delta_x, delta_y, delta_z], robot, sim_params.dt, 3, vel_limits=vel_limits)
            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            state = "move to goal"


        if state == "move to goal":

            if num_new_goal >= max_num_new_goal \
                    or timeit.default_timer() - plane_start_time >= max_plane_time:
                print(num_new_goal, num_new_goal >= max_num_new_goal, timeit.default_timer() - plane_start_time >= max_plane_time)
                rospy.logerr("Timeout")
                state = "reset"   
                

            main_ins_pos = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"))
            if main_ins_pos <= 0.042:
                rospy.logerr("Exceed joint constraint")
                state = "reset"


            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
                rospy.logerr("Lost contact with robot")
                state = "reset"
                if num_new_goal == 0:
                    all_done = True

            else:
                if frame_count % 15 == 0: #15
                    full_pc_on_trajectory.append(get_point_cloud())
                    pc_on_trajectory.append(get_partial_point_cloud(obj_name="tissue"))
                    

                    terminate_count += 1
                    if terminate_count >= 15:
                        print("max terminate count")
                        state = "reset"
                        terminate_count = 0
                frame_count += 1   
     
                
             
                action = tvc_behavior.get_action()  
                if action is not None and gym.get_sim_time(sim) - closed_loop_start_time <= 8: 
                    gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())
                else:   
                    rospy.loginfo("Succesfully executed moveit arm plan. Let's record point cloud!!")  

                    if visualization:
                        pcd = pcd_ize(get_partial_point_cloud(obj_name="tissue"))
                        pcd.paint_uniform_color([0,0,0])
                        open3d.visualization.draw_geometries([pcd, pcd_plane, pcd_kidney])

                    success = check_success(constrain_plane, current_pc=get_point_cloud())
                   
                    if not success:
                        num_new_goal += 1
                        rospy.logerr(f"Got a new goal. num_new_goal: {num_new_goal}")
                        state = "get shape servo plan"   
                        all_done = True #FIX                     
   
                    else:
                        
                        rospy.logerr("Succeeded task !!!")
                        
                        pc_goal = get_partial_point_cloud(obj_name="tissue")
                        full_pc_goal = get_point_cloud()
                        
                        for j in range(len(pc_on_trajectory)):                                                    
                            partial_pcs = (pc_on_trajectory[j], pc_goal)
                            full_pcs = (full_pc_on_trajectory[j], full_pc_goal)


                            data = {"full pcs": full_pcs, "partial pcs": partial_pcs, "constrain_plane": constrain_plane,\
                                    "kidney_pc": kidney_pc, "kidney_idx": args.kidney_idx, "angles": (kidney_angle, tissue_angle)}
                            with open(os.path.join(data_recording_path, "sample " + str(data_point_count) + ".pickle"), 'wb') as handle:
                                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)                               
                            print("data_point_count:", data_point_count)
                            
                            data_point_count += 1   
                               

                        
                        frame_count = 0
                        terminate_count = 0
                        state = "reset" 


                    
                    

        if state == "reset":   
            rospy.loginfo("**Current state: " + state)
            frame_count = 0
            # sample_count = 0
            terminate_count = 0
            num_new_goal = 0
            sample_count += 1
            first_action = True


            
            dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_POS)
            dof_props_2["stiffness"][:8].fill(200.0)
            dof_props_2["damping"][:8].fill(40.0)
            

            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], init_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, [0]*8)
            
            print("Sucessfully reset robot and object")
            pc_on_trajectory = []
            full_pc_on_trajectory = []
            # curr_trans_on_trajectory = []
                

            gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)  
            gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, [0,0,0,0,0.22,0,0,0,1.5,0.8]) 


            state = "home"
 

        # if  data_point_count >= max_data_point_count:       
        if sample_count >= max_sample_count:             
            all_done = True 

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
    print("total data pt count: ", data_point_count)
