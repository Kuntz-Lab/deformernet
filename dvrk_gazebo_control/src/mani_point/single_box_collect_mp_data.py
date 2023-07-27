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
from utils.isaac_utils import fix_object_frame, get_pykdl_client
# from utils.record_data_h5 import RecordGraspData_sparse
import pickle
# from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
from PIL import Image
import transformations
from chamferdist import ChamferDistance

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl


sys.path.append('/home/baothach/shape_servo_DNN/generalization_tasks')
# from pointcloud_recon_2 import PointNetShapeServo, PointNetShapeServo2
from architecture import DeformerNet
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

def down_sampling(pc):
    farthest_indices,_ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]  
    return pc


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
    # soft_asset_file = "cheez_it_white_cheddar.urdf"
    soft_asset_file = "box.urdf"

    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.0, -0.42, 0.01818)
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.001#0.001   # important to add some thickness to the soft body to avoid interpenetrations






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
        cam_pos = gymapi.Vec3(1, 0.5, 1)
        # cam_pos = gymapi.Vec3(0.3, -0.7, 0.3)
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
    cam_positions.append(gymapi.Vec3(0.1, -0.5, 0.2))
    cam_targets.append(gymapi.Vec3(0.0, -0.45, 0.00))
  

    
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
    


    data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/box/data"
    mp_classifer_data_path = "/home/baothach/shape_servo_data/manipulation_points/box/mp_classifer_data" 
    log_dir =  "/home/baothach/shape_servo_data/manipulation_points/box/logs"
    first_time = True
    frame_count = 0
    sample_count = 112
    mp_count = 1
    max_sample_count = 1000
    max_mp_count = 11 # 1 for ground truth and 10 random MPs
    num_recorded_mp_data = 9379

    min_chamfer_dist = 0.2   
    max_shapesrv_time = 1*60    # 1 mins 
    fail_mtp = False
    gt_chamfer = None
    saved_chamfers = []
    final_chamfer_distances = []   
    recorded_mps = []
    final_pcs = []
    all_chamfers = [] 


    # Sample some init-goal pc pairs from the data
    sampled_idxs = np.random.randint(low=0, high=num_recorded_mp_data, size=max_sample_count)

    dc_client = GraspDataCollectionClient()
    chamferDist = ChamferDistance()

    # Set up DNN:
    device = torch.device("cuda")
    model = DeformerNet(normal_channel=False)
    
    # if args.obj_type == "box_1k":
    weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_1000Pa/weights/run2(partial)"
    model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 240")))      




    
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
            if frame_count == 5:
                rospy.loginfo("**Current state: " + state)
                rospy.logwarn(f"sample {sample_count}; mp {mp_count}")
                

                if first_time:                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                    first_time = False
                
                # state = "generate preshape"
                
                desired_position = np.array([0.,0.,0.]) # Set intiial desired gripper position
                # frame_count = 0

                initial_pc = get_point_cloud()
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(initial_pc))
                open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                pc_ros_msg = fix_object_frame(pc_ros_msg)


                if mp_count == 0:
                    # Get goal pc:
                    with open(os.path.join(data_recording_path, "sample " + str(sampled_idxs[sample_count]) + ".pickle"), 'rb') as handle:
                        data = pickle.load(handle)
                        goal_pc_numpy = down_sampling(data["partial pcs"][1])
                        goal_pc_tensor = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float() 
                        init_pc_numpy = down_sampling(data["partial pcs"][0])
                        init_pc_tensor = torch.from_numpy(init_pc_numpy).unsqueeze(0).float()    
       

                    print("check chamfer:", np.sqrt(chamferDist(goal_pc_tensor.permute(0,2,1), init_pc_tensor).item()))
                    if np.sqrt(chamferDist(goal_pc_tensor.permute(0,2,1), init_pc_tensor).item()) <= 0.2:   # init and goal pc too close
                        rospy.logerr("init and goal pc too close. Move on to next sample")
                        sampled_idxs[sample_count] = np.random.randint(low=0, high=num_recorded_mp_data, size=1)
                        frame_count -= 1    # redo with a new sample
                        state = "home"

                    else:
                        pcd_goal = open3d.geometry.PointCloud()
                        pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
                        # goal_position = data["positions"]  
                        saved_obj_contact_state = data["obj contact state"]
                        saved_robot_contact_state = data["robot contact state"]
                        recorded_mps.append(data["mani_point"]["pose"])
  
                       



                        gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], saved_robot_contact_state, gymapi.STATE_ALL) 
                        gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_obj_contact_state))
                        gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), 0.35)
                        gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), -0.35)  
                        dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                        dof_props_2["stiffness"][:8].fill(0.0)
                        dof_props_2["damping"][:8].fill(200.0)
                        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)
                        
                        shapesrv_start_time = timeit.default_timer()
                        # state = "get shape servo plan"
            elif frame_count == 6:
                if mp_count == 0:  
                    state = "get shape servo plan"
                else:
                    state = "generate preshape"              
                frame_count = 0

        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
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
        
                # current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])
                # print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] )
                anchor_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])

                # switch to velocity mode
                # dof_props_2 = gym.get_asset_dof_properties(kuka_asset)
                dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                # dof_props_2["driveMode"][8:].fill(gymapi.DOF_MODE_POS)
                dof_props_2["stiffness"][:8].fill(0.0)
                dof_props_2["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

                # Get current MP
                recorded_mps.append(anchor_pose["pose"])
                shapesrv_start_time = timeit.default_timer()


        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state)

            current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])
            # print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] ) 

            current_pc = down_sampling(get_partial_point_cloud(i))                  
                            
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(current_pc)  
            # open3d.visualization.draw_geometries([pcd, pcd_goal])  
            chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
            saved_chamfers.append(chamfer_dist)
            print("chamfer distance: ", chamfer_dist)

            if chamfer_dist >= min_chamfer_dist:
                current_pc = torch.from_numpy(current_pc).permute(1,0).unsqueeze(0).float()
                with torch.no_grad():
                    desired_position = model(current_pc, goal_pc_tensor)[0].detach().numpy()*(0.001) 
            print("from model:", desired_position)
            # print("ground truth: ", goal_position)             
            # delta_x = desired_position[0]   
            # delta_y = desired_position[1] 
            # delta_z = desired_position[2] 

            tvc_behavior = TaskVelocityControl(list(desired_position), robot, sim_params.dt, 3, vel_limits=vel_limits, error_threshold = 3e-3)     

            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            # test = timeit.default_timer()

            state = "move to goal"


        if state == "move to goal":           
            # rospy.loginfo("**Current state: " + state)
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
                recorded_mps.pop()
                rospy.logerr("Lost contact with robot")
                state = "reset" 

            else:
                if timeit.default_timer() - shapesrv_start_time >= max_shapesrv_time:
                    rospy.logerr("Timeout")
                    state = "reset" 
                    current_pc = get_partial_point_cloud(i)
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(current_pc)   
                    chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                    
                    saved_chamfers.append(chamfer_dist)
                    final_chamfer_distances.append(min(saved_chamfers)) 
                    # final_pcs.append(down_sampling(get_partial_point_cloud(i))) 
                    
                    if gt_chamfer is None:
                        assert mp_count == 0
                        gt_chamfer =  min(saved_chamfers)

                    mp_count += 1

                else:
                    action = tvc_behavior.get_action()  
                    if action is None or gym.get_sim_time(sim) - closed_loop_start_time >= 3:#1.5:   
                       
                        state = "get shape servo plan"    
                    else:
                        # print("desired vel:", action.get_joint_position())
                        gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())        

                    # Terminal conditions
                    if all(abs(desired_position) <= 0.005) \
                            or chamfer_dist < min_chamfer_dist:
                        
                        current_pc = current_pc = get_partial_point_cloud(i)
                        pcd = open3d.geometry.PointCloud()
                        pcd.points = open3d.utility.Vector3dVector(current_pc)  
                        chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                        print("final chamfer distance: ", chamfer_dist)
                        final_chamfer_distances.append(chamfer_dist) 
                        if gt_chamfer is None:
                            assert mp_count == 0
                            gt_chamfer =  chamfer_dist
                        
                        
                        # final_pcs.append(down_sampling(get_partial_point_cloud(i)))                        
                        mp_count += 1
                        
                        
                        state = "reset" 



        if state == "reset":   
            rospy.loginfo("**Current state: " + state)
            print("Total elapsed time", timeit.default_timer() - start_time)
            print(f"time passed sample {sample_count} mp {mp_count}: {timeit.default_timer() - shapesrv_start_time}s")
            frame_count = 0
            saved_chamfers = []
            
            rospy.logwarn(f"=== JUST ENDED sample {sample_count} mp {mp_count-1}")           



            if mp_count == max_mp_count:
   
                 
                all_chamfers.extend(final_chamfer_distances)
                # record data 
                if len(final_chamfer_distances) != max_mp_count or  len(recorded_mps) != max_mp_count:
                    rospy.logerr(f"incorrect lengths: {len(final_chamfer_distances)}, {len(recorded_mps)}")
                assert len(final_chamfer_distances) == max_mp_count
                assert len(recorded_mps) == max_mp_count  
                
                    
                for idx in range(max_mp_count):
                    pcs = data["partial pcs"]
                    mani_point = recorded_mps[idx] # check this
                    final_chamfer = final_chamfer_distances[idx]
                    # final_pc = final_pcs[idx]
                    gt_mp = recorded_mps[0] # check this
                    gt_chamfer = final_chamfer_distances[0]                    

                    final_data = {"partial pcs": data["partial pcs"], "mani_point": mani_point, "chamfer": final_chamfer, \
                                   "gt mani_point": gt_mp, "gt chamfer": gt_chamfer, "obj_name": args.obj_name}                                                                   
                    # final_data = {"partial pcs": data["partial pcs"], "mani_point": mani_point, "chamfer": final_chamfer, \
                    #                "final pc": final_pc, "gt mani_point": gt_mp, "gt chamfer": gt_chamfer, "obj_name": args.obj_name}  

                    with open(os.path.join(mp_classifer_data_path, f"sample {sample_count} mp {idx}.pickle"), 'wb') as handle:
                        pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)      

                gt_chamfer = None
                mp_count = 0
                sample_count += 1  
                final_chamfer_distances = []
                recorded_mps = []
                final_pcs = []


            if fail_mtp:
                state = "home"  
                fail_mtp = False

            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], init_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, [0]*8)
            dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_POS)
            dof_props_2["stiffness"][:8].fill(200.0)
            dof_props_2["damping"][:8].fill(40.0)
            gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

            print("Sucessfully reset robot and object")
            shapesrv_start_time = timeit.default_timer()

            state = "home"

        if  sample_count >= max_sample_count:                    
            all_done = True 



        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)



    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


    all_chamfers.extend(final_chamfer_distances)
    print(all_chamfers)
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        print(f"Chamfer < {threshold}m count:", sum(1 for chamf in all_chamfers if chamf <= threshold))
    
    print("===========")
    
    for threshold in [0.8, 1.0]:
        print(f"Chamfer > {threshold}m count:", sum(1 for chamf in all_chamfers if chamf >= threshold))  


    writer = open(os.path.join(log_dir, f'logs_{len(os.listdir(log_dir))}.txt'), 'w')  
    writer.write(f"Data path:{mp_classifer_data_path}\n")   
    writer.write("\n")
    writer.write(f"All Chamfer distances:{all_chamfers}\n")  
    writer.write("\n")
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        writer.write(f"Chamfer < {threshold}m count: {sum(1 for chamf in all_chamfers if chamf <= threshold)}\n")
    
    writer.write("\n")
    
    for threshold in [0.8, 1.0]:
        writer.write(f"Chamfer > {threshold}m count: {sum(1 for chamf in all_chamfers if chamf >= threshold)}\n")  