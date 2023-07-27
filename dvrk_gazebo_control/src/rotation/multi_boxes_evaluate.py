#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys

from numpy import random
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
from utils.isaac_utils import fix_object_frame, get_pykdl_client, pad_rot_mat
# from utils.record_data_h5 import RecordGraspData_sparse
import pickle
# from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
from PIL import Image
import transformations
from sklearn.neighbors import NearestNeighbors

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl2


# sys.path.append('/home/baothach/shape_servo_DNN/rotation')
# from architecture_3 import DeformerNetMP as DeformerNet
# sys.path.remove('/home/baothach/shape_servo_DNN/rotation')
sys.path.append('/home/baothach/shape_servo_DNN/generalization_tasks')
from architecture import DeformerNet as DeformerNet2
sys.path.remove('/home/baothach/shape_servo_DNN/generalization_tasks')


import torch


ROBOT_Z_OFFSET = 0.25
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 0.86

sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *



def init():
    for i in range(num_envs):
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
            {"name": "--obj_name", "type": str, "default": 'box_0', "help": "select variations of a primitive shape"},
            {"name": "--headless", "type": bool, "default": False, "help": "headless mode"}])

    num_envs = args.num_envs
    


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
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 50
        sim_params.flex.relaxation = 0.7
        sim_params.flex.warm_start = 0.1
        sim_params.flex.shape_collision_distance = 5e-4
        sim_params.flex.contact_regularization = 1.0e-6
        sim_params.flex.shape_collision_margin = 1.0e-4
        sim_params.flex.deterministic_mode = True
        # sim_params.flex.dynamic_friction = 70
        # sim_params.flex.static_friction = 100        

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)
    print("==========args.compute_device_id:", args.compute_device_id)
    # # Get primitive shape dictionary to know the dimension of the object   
    # object_meshes_path = "/home/baothach/sim_data/Custom/Custom_mesh/multi_boxes_5kPa"    
    # with open(os.path.join(object_meshes_path, "primitive_dict_box.pickle"), 'rb') as handle:
    #     data = pickle.load(handle)    
    # h = data[args.obj_name]["height"]
    # w = data[args.obj_name]["width"]
    # thickness = data[args.obj_name]["thickness"]

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
    asset_options.thickness = 0.0005
    # asset_options.thickness = 0.002


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

    asset_root = "/home/baothach/sim_data/Custom/Custom_urdf"
    soft_asset_file = "box.urdf"
    # soft_asset_file = "smaller_box.urdf"

    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.0, -0.42, 0.01818)
    # soft_pose.p = gymapi.Vec3(0.0, -0.42, 0.01)
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.001#0.001   # important to add some thickness to the soft body to avoid interpenetrations
    # soft_thickness = 0.0005






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
        cam_pos = gymapi.Vec3(0.3, -0.7, 0.3)
        # cam_pos = gymapi.Vec3(0.3, -0.1, 0.5)  # final setup for thin layer tissue
        # cam_pos = gymapi.Vec3(0.5, -0.36, 0.3)
        # cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
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
    # cam_positions.append(gymapi.Vec3(0.1, -0.5, 0.2))
    # cam_targets.append(gymapi.Vec3(0.0, -0.45, 0.00))
    cam_positions.append(gymapi.Vec3(0.17, -0.62, 0.2))
    cam_targets.append(gymapi.Vec3(0.0, 0.40-two_robot_offset, 0.01))    

    
    for i, env_obj in enumerate(envs_obj):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])
    print("============cam_handle:", cam_handles[0])



    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

        

    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')
    rospy.logerr("======Loading object ... " + str(args.obj_name))  
 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    

    # goal_recording_path = "/home/baothach/shape_servo_data/comparison/RRT/goal_data"
    

    data_recording_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/goal_data/with_rotation"
    terminate_count = 0
    sample_count = 0
    frame_count = 0
    group_count = 0
    data_point_count = 5098
    max_group_count = 1500
    max_sample_count = 4
    max_data_point_count = 10000
    random_count = 0
    # if args.obj_name == 'box_64':
    #     max_data_point_per_variation = 9600
    # else:
    #     max_data_point_per_variation = data_point_count + 150
    # rospy.logwarn("max_data_point_per_variation:" + str(max_data_point_per_variation))

    final_point_clouds = []
    final_desired_positions = []
    pc_on_trajectory = []
    full_pc_on_trajectory = []
    curr_trans_on_trajectory = []
    first_time = True
    save_intial_pc = True
    switch = True
    total_computation_time = 0
    data = []

    dc_client = GraspDataCollectionClient()

    goal_point_count = 209
    execute_count = 0
    max_goal_point_count = 1000#goal_point_count+1
    with_init_pose = True
    set_init_pose = False
    with_rotation =True
    use_gripper_eulers =True

    vis = True
    dual_vis = False
    meta_model = True


    # Set up DNN:
    device = torch.device("cuda")
    # model = DeformerNet(normal_channel=False).to(device)
    # weight_path = "/home/baothach/shape_servo_data/rotation_extension/box/weights/run2/"

    if meta_model:
        if use_gripper_eulers:
            sys.path.append('/home/baothach/shape_servo_DNN/rotation')
            from architecture_3 import DeformerNetMP as DeformerNet
            model = DeformerNet(normal_channel=False).to(device)
            weight_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/weights/run3(gripper_eulers)/"
            model.load_state_dict(torch.load(weight_path + "epoch " + str(142)))            
        else:
            sys.path.append('/home/baothach/shape_servo_DNN/rotation')
            from architecture_2 import DeformerNetMP as DeformerNet
            model = DeformerNet(normal_channel=False).to(device)
            weight_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/weights/run1/"
            model.load_state_dict(torch.load(weight_path + "run1epoch " + str(150)))

    # else:
    #     weight_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/weights/old_model/"
    #     model.load_state_dict(torch.load(weight_path + "epoch " + str(240)))

        # weight_path = "/home/baothach/shape_servo_data/rotation_extension/box/weights/mp/run2(smaller_obj)/"
        # model.load_state_dict(torch.load(weight_path + "epoch " + str(150)))
    model.eval()

    model_no_rot = DeformerNet2(normal_channel=False).to(device)
    weight_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/weights/old_model/"
    model_no_rot.load_state_dict(torch.load(weight_path + "epoch " + str(240)))
    model_no_rot.eval()

    

    # Get goal pc:
    with open(os.path.join(data_recording_path, "sample " + str(goal_point_count) + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)
        goal_pc_numpy = data["partial pc"]
        farthest_indices,_ = farthest_point_sampling(goal_pc_numpy, 1024)
        goal_pc_numpy = goal_pc_numpy[farthest_indices.squeeze()]

        goal_pc = torch.from_numpy(np.swapaxes(goal_pc_numpy,0,1)).float().to(device)
        pcd_goal = open3d.geometry.PointCloud()
        pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
        pcd_goal.paint_uniform_color([1, 0, 0]) 
        goal_pos = data["pos"] 
        goal_rot = data["rot"] 

    # with open(os.path.join(data_recording_path, "sample " + str(100) + ".pickle"), 'rb') as handle:
    #     datax = pickle.load(handle)
    #     goal_pc_numpyx = datax["partial pc"]
    #     pcd_goalx = open3d.geometry.PointCloud()
    #     pcd_goalx.points = open3d.utility.Vector3dVector(goal_pc_numpyx) 
    #     pcd_goalx.paint_uniform_color([0, 0, 1]) 
    
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
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.203)            
            if frame_count == 10:
                rospy.loginfo("**Current state: " + state + ", current sample count: " + str(sample_count))
                

                if first_time:                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                    first_time = False

                    current_pc = get_point_cloud()
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                    open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                    pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                    pc_ros_msg = fix_object_frame(pc_ros_msg)
                
                state = "generate preshape"                
                frame_count = 0              


        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
            cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            target_pose = [-cartesian_goal.position.x, -cartesian_goal.position.y, cartesian_goal.position.z-ROBOT_Z_OFFSET-0.03,
                            0, 0.707107, 0.707107, 0]
            # euler = list(transformations.euler_from_quaternion([0, 0.707107, 0.707107,0 ]))
            # euler[0] += np.pi/2
            # quat = transformations.quaternion_from_euler(*euler)
            # target_pose = [-cartesian_goal.position.x, -cartesian_goal.position.y, cartesian_goal.position.z-ROBOT_Z_OFFSET-0.03,
            #                 *quat ]
            
            # print("****test:", test)
            mtp_behavior = MoveToPose(target_pose, robot, sim_params.dt, 1)
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

                anchor_pose = deepcopy(init_pose)
                anchor_eulers = deepcopy(init_eulers)


                dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                dof_props_2["stiffness"][:8].fill(0.0)
                dof_props_2["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

        if state == "get shape servo plan":
            # with_rotation = False
            if set_init_pose and with_init_pose:
                delta_x, delta_y, delta_z = -0.06,0.06,0.06
                delta_alpha, delta_beta, delta_gamma = -np.pi/40000,-np.pi/40000,-np.pi/40000
                # delta_x, delta_y, delta_z = -0.04,0.04,0.04
                # delta_alpha, delta_beta, delta_gamma = -np.pi/3,-np.pi/3,-np.pi/3 
                x = delta_x + init_pose[0,3]
                y = delta_y + init_pose[1,3]
                z = delta_z + init_pose[2,3]
                alpha = delta_alpha + init_eulers[0]
                beta = delta_beta + init_eulers[1]
                gamma = delta_gamma + init_eulers[2]
            
                tvc_behavior = TaskVelocityControl2([x,y,z,alpha,beta,gamma], robot, sim_params.dt, 3, vel_limits=vel_limits)      
                set_init_pose = False                      
            
            elif not with_rotation:
                current_pc = get_partial_point_cloud(i)
                farthest_indices,_ = farthest_point_sampling(current_pc, 1024)
                current_pc_numpy = current_pc[farthest_indices.squeeze()]                 
                current_pc = torch.from_numpy(np.swapaxes(current_pc,0,1)).float().to(device)
                desired_pos  = model_no_rot(current_pc.unsqueeze(0), goal_pc.unsqueeze(0))[0].detach().cpu().numpy()*(0.001)
                print("desired_pos:", desired_pos)
                desired_pos = (desired_pos + init_pose[:3,3]).flatten()
                delta_alpha, delta_beta, delta_gamma = np.pi/3000000,np.pi/3000000,-np.pi/30000000
                alpha = delta_alpha + init_eulers[0]
                beta = delta_beta + init_eulers[1]
                gamma = delta_gamma + init_eulers[2]
            
                tvc_behavior = TaskVelocityControl2([*desired_pos,alpha,beta,gamma], robot, sim_params.dt, 3, vel_limits=vel_limits,\
                                                    pos_threshold = 2e-3, ori_threshold=5e-2)      
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(current_pc_numpy)  
                # open3d.visualization.draw_geometries([pcd, pcd_goal])  
                chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                print("chamfer distance: ", chamfer_dist)
                # print("chamfer distancex: ", np.linalg.norm(np.asarray(pcd_goalx.compute_point_cloud_distance(pcd))))
                print("================")      
            else:
                current_pc = get_partial_point_cloud(i)
                farthest_indices,_ = farthest_point_sampling(current_pc, 1024)
                current_pc_numpy = current_pc[farthest_indices.squeeze()]    
                # current_pc = torch.from_numpy(current_pc).permute(1,0).float().to(device)

                mani_point = init_pose[:3,3] * np.array([-1,-1,1]) + np.array([0,0, ROBOT_Z_OFFSET])
                neigh = NearestNeighbors(n_neighbors=50)
                neigh.fit(current_pc_numpy)
                _, nearest_idxs = neigh.kneighbors(mani_point.reshape(1, -1))
                mp_channel = np.zeros(current_pc_numpy.shape[0])
                mp_channel[nearest_idxs.flatten()] = 1
                modified_pc = np.vstack([current_pc_numpy.transpose(1,0), mp_channel])
                current_pc = torch.from_numpy(modified_pc).float().to(device)

                if use_gripper_eulers:
                    gripper_eulers = torch.from_numpy(np.array(init_eulers)).unsqueeze(0).float().to(device)
                    pos, rot_mat = model(current_pc.unsqueeze(0), goal_pc.unsqueeze(0), gripper_eulers)
                else:    
                    pos, rot_mat = model(current_pc.unsqueeze(0), goal_pc.unsqueeze(0))
  
                pos *= 0.001
                pos, rot_mat = pos.detach().cpu().numpy(), rot_mat.detach().cpu().numpy()
                desired_pos = (pos + init_pose[:3,3]).flatten()
                # desired_rot = rot_mat @ np.linalg.inv(init_pose[:3,:3])
                desired_rot = rot_mat @ init_pose[:3,:3]
                # print("curr rot:", transformations.euler_from_matrix(get_pykdl_client(robot.get_arm_joint_positions())[1]))
                # print("desired rot:", transformations.euler_from_matrix(pad_rot_mat(desired_rot)))
        

                temp1 = np.eye(4)
                temp1[:3,:3] = rot_mat
                temp2 = np.eye(4)
                temp2[:3,:3] = goal_rot            
                print("pos, rot_mat:", pos, transformations.euler_from_matrix(temp1))
                print("goal_pos, goal_rot:", goal_pos, transformations.euler_from_matrix(temp2))


                tvc_behavior = TaskVelocityControl2([*desired_pos, desired_rot], robot, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                    pos_threshold = 2e-3, ori_threshold=5e-2)
                # print("current delta eulers:", (np.array(init_eulers) - np.array(anchor_eulers))/np.pi)

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(current_pc_numpy)  
                # open3d.visualization.draw_geometries([pcd, pcd_goal])  
                chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                print("chamfer distance: ", chamfer_dist)
                # print("chamfer distancex: ", np.linalg.norm(np.asarray(pcd_goalx.compute_point_cloud_distance(pcd))))
                print("================")

            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            state = "move to goal"

        if state == "move to goal":           
            action = tvc_behavior.get_action()  
            # print("moving")
            # print("time, action, complete:", timeit.default_timer() - closed_loop_start_time, action, mtp_behavior.is_complete())
            if action is None or gym.get_sim_time(sim) - closed_loop_start_time >= 7:   
                # final_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])
                # print("***Final x, y, z: ", final_pose["pose"]["p"]["x"], final_pose["pose"]["p"]["y"], final_pose["pose"]["p"]["z"] ) 
                # # delta_x = -(final_pose["pose"]["p"]["x"] - anchor_pose["pose"]["p"]["x"])
                # # delta_y = -(final_pose["pose"]["p"]["y"] - anchor_pose["pose"]["p"]["y"])
                # # delta_z = final_pose["pose"]["p"]["z"] - anchor_pose["pose"]["p"]["z"]
                # # print("delta x, y, z:", delta_x, delta_y, delta_z)
                state = "get shape servo plan"  
                execute_count += 1
                if execute_count >= 4 or chamfer_dist <= 0.20:
                    state = "reset" 
                _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                init_eulers = transformations.euler_from_matrix(init_pose)                

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(get_partial_point_cloud(i)) 
                pcd.paint_uniform_color([0, 0, 0])                 
                if vis:
                    if dual_vis:
                        open3d.visualization.draw_geometries([pcd, pcd_goal, pcd_goal]) 
                    else:
                        # mani_point_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                        # mani_point_sphere.paint_uniform_color([0,0,1])

                        # open3d.visualization.draw_geometries([pcd, pcd_goal, mani_point_sphere.translate((mani_point[0], mani_point[1], mani_point[2]))])
                        open3d.visualization.draw_geometries([pcd, pcd_goal])
            else:
                gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())                


        if state == "reset":   
            rospy.loginfo("**Current state: " + state)
            frame_count = 0
            sample_count = 0
            execute_count = 0

            
            dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_POS)
            dof_props_2["stiffness"][:8].fill(200.0)
            dof_props_2["damping"][:8].fill(40.0)
            gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)
            

            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], init_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, [0]*8)
            
            print("Sucessfully reset robot and object")
            
            goal_point_count += 1
            print("***goal_point_count:", goal_point_count)
            with open(os.path.join(data_recording_path, "sample " + str(goal_point_count) + ".pickle"), 'rb') as handle:
                data = pickle.load(handle)
                goal_pc_numpy = data["partial pc"]
                farthest_indices,_ = farthest_point_sampling(goal_pc_numpy, 1024)
                goal_pc_numpy = goal_pc_numpy[farthest_indices.squeeze()]

                goal_pc = torch.from_numpy(np.swapaxes(goal_pc_numpy,0,1)).float().to(device)
                pcd_goal = open3d.geometry.PointCloud()
                pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
                pcd_goal.paint_uniform_color([1, 0, 0]) 
                goal_pos = data["pos"] 
                goal_rot = data["rot"] 
                

            


            state = "home"
 
        
        if sample_count == max_sample_count:  
            sample_count = 0            
            group_count += 1
            print("group count: ", group_count)
            state = "reset" 



        # if group_count == max_group_count or data_point_count >= max_data_point_count: 
        if goal_point_count >= max_goal_point_count:           
            all_done = True 

        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)
            # gym.sync_frame_time(sim)


  
    # with open(os.path.join(pkg_path, "src/rotation/debug/combine.pickle"), 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)      



    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("total data pt count: ", data_point_count)