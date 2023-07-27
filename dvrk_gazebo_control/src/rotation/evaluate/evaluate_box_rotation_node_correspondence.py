#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys
# from turtle import width

from numpy.lib.polynomial import polyint
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
# from PIL import Image
import transformations
from sklearn.neighbors import NearestNeighbors

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl, TaskVelocityControl2



sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *

import torch



ROBOT_Z_OFFSET = 0.25
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 0.86



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

def down_sampling(pc):
    farthest_indices,_ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]  
    return pc


if __name__ == "__main__":

    main_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/evaluate"
    objects_path = "/home/baothach/shape_servo_data/evaluation"

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--obj_name", "type": str, "default": 'box_0', "help": "select variations of a primitive shape"},
            {"name": "--obj_type", "type": str, "default": 'box_1k', "help": "box1k, box5k, etc."},
            {"name": "--inside", "type": str, "default": "True", "help": "inside train distribution"},
            {"name": "--no_rot", "type": str, "default": "False", "help": "use model with orientation"},
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"}])

    num_envs = args.num_envs
    
    args.headless = args.headless == "True"
    args.inside = args.inside == "True"
    args.no_rot = args.no_rot == "True"

    


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
        sim_params.flex.num_outer_iterations = 10#4
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
        object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "inside_sample_ratio")
    else:
        object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "outside") 

    with open(os.path.join(object_meshes_path, args.obj_name + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)    
    h = data["height"]
    w = data["width"]
    thickness = data["thickness"]

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
        asset_root = os.path.join(objects_path, "urdf", args.obj_type, "inside_sample_ratio")
    else:
        asset_root = os.path.join(objects_path, "urdf", args.obj_type, "outside") 


    soft_asset_file = args.obj_name + ".urdf"    
    # asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/test"
    # soft_asset_file = "long_box.urdf"
    # asset_root = "/home/baothach/Downloads"
    # soft_asset_file = "test_box.urdf"



    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.0, -0.42, thickness/2*0.7)
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
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
 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    first_time = True    


    # Set up DNN:
    device = torch.device("cuda")
    
    
    no_rot = args.no_rot #False#True
    if args.obj_type == "box_1k":
        if no_rot:
            sys.path.append('/home/baothach/shape_servo_DNN/generalization_tasks')
            from architecture import DeformerNet            
            model = DeformerNet(normal_channel=False).to(device)
            weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_1000Pa/weights/run2(partial)"
            model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 240")))             
        else:
            sys.path.append('/home/baothach/shape_servo_DNN/rotation')
            from architecture_2 import DeformerNetMP as DeformerNet
            model = DeformerNet(normal_channel=False).to(device)
            # weight_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa/weights/run1/"
            # model.load_state_dict(torch.load(os.path.join(weight_path, "run1epoch 150")))  
            weight_path = "/home/baothach/shape_servo_data/rotation_extension/multi_boxes_1000Pa_2/weights/run1/"
            model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 524")))  
    # elif args.obj_type == "box_5k":
    #     # weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_5kPa/weights/run1"
    #     # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 250")))  
    #     weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_5kPa/weights/run3(partial)"
    #     model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 240")))              
    # elif args.obj_type == "box_10k":
    #     weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_10kPa/weights/run1"
    #     model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 288"))) 

    model.eval()


    if args.inside:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "inside")
        # chamfer_recording_path = os.path.join(main_path, "chamfer_results", args.obj_type, "inside")
        if args.no_rot:
            chamfer_recording_path = os.path.join(main_path, "chamfer_results", args.obj_type, "inside_sample_ratio_no_rot")
        else:
            chamfer_recording_path = os.path.join(main_path, "chamfer_results", args.obj_type, "inside_sample_ratio_w_rot")
    else:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "outside") 
        chamfer_recording_path = os.path.join(main_path, "chamfer_results", args.obj_type, "outside")

    goal_count = 2#0 #0
    frame_count = 0
    max_goal_count =  10  #10

    max_shapesrv_time = 1.5*60    # 2 mins
    if args.inside:
        min_chamfer_dist = 0.1 #0.2
    else:
        min_chamfer_dist = 0.2 #0.25
    fail_mtp = False
    saved_nodes = []
    saved_chamfers = []
    final_node_distances = []  
    final_chamfer_distances = []    

    dc_client = GraspDataCollectionClient()   

   
    # Get 10 goal pc data for 1 object:
    with open(os.path.join(goal_recording_path, args.obj_name + "_test.pickle"), 'rb') as handle:
        goal_datas = pickle.load(handle) 
    goal_pc_numpy = down_sampling(goal_datas[goal_count]["partial pcs"][1])   # first goal pc
    goal_pc_tensor = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device) 
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy)  
    pcd_goal.paint_uniform_color([1,0,0]) 

    full_pc_goal = goal_datas[goal_count]["full pcs"][1]
    rospy.logwarn(f"number of nodes on mesh: {full_pc_goal.shape}")


    if no_rot:
        goal_position = goal_datas[goal_count]["pos"].T.squeeze()
    else:
        goal_pos = goal_datas[goal_count]["pos"] 
        goal_rot = goal_datas[goal_count]["rot"] 
    saved_obj_contact_state = goal_datas[goal_count]["obj contact state"]
    saved_robot_contact_state = goal_datas[goal_count]["robot contact state"]
    print("=========vel_limits",vel_limits)
    
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
            if frame_count == 1:
                rospy.loginfo("**Current state: " + state)
                

                # if first_time:                    
                #     gym.refresh_particle_state_tensor(sim)
                #     saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                #     init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                #     first_time = False
                
                # state = "get shape servo plan"
                
                desired_position = np.array([0.,0.,0.]) # Set intiial desired gripper position
                # frame_count = 0

                gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], saved_robot_contact_state, gymapi.STATE_ALL) 
                gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_obj_contact_state))
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), 0.35)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), -0.35)  
                dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                dof_props_2["stiffness"][:8].fill(0.0)
                dof_props_2["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)
                
                shapesrv_start_time = timeit.default_timer()
                # open3d.visualization.draw_geometries([pcd_goal])


                # anchor_pose = deepcopy(init_pose)
                # anchor_eulers = deepcopy(init_eulers)    


            elif frame_count == 6:
                frame_count = 0
                _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                init_eulers = transformations.euler_from_matrix(init_pose)
                state = "get shape servo plan"



        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state)

            current_pc_numpy = down_sampling(get_partial_point_cloud(i))                  
                            
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(current_pc_numpy)  
            pcd.paint_uniform_color([0,0,0])
            # open3d.visualization.draw_geometries([pcd, pcd_goal]) 


            node_dist = np.linalg.norm(full_pc_goal - get_point_cloud())
            saved_nodes.append(node_dist)
            rospy.logwarn(f"Node distance: {node_dist}")

            chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
            saved_chamfers.append(chamfer_dist)
            rospy.logwarn(f"chamfer distance: {chamfer_dist}")
        
            
            

            if no_rot:
                current_pc_tensor = torch.from_numpy(current_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)
            else:
                mani_point = init_pose[:3,3] * np.array([-1,-1,1]) + np.array([0,0, ROBOT_Z_OFFSET])
                neigh = NearestNeighbors(n_neighbors=50)
                neigh.fit(current_pc_numpy)
                _, nearest_idxs = neigh.kneighbors(mani_point.reshape(1, -1))
                mp_channel = np.zeros(current_pc_numpy.shape[0])
                mp_channel[nearest_idxs.flatten()] = 1
                modified_pc = np.vstack([current_pc_numpy.transpose(1,0), mp_channel])
                current_pc_tensor = torch.from_numpy(modified_pc).unsqueeze(0).float().to(device)
                # print("current_pc_tensor.shape, goal_pc_tensor.shape:", current_pc_tensor.shape, goal_pc_tensor.shape)

            if True: #chamfer_dist >= min_chamfer_dist:
                with torch.no_grad():
                    if no_rot:
                        desired_position = model(current_pc_tensor, goal_pc_tensor)[0].cpu().detach().numpy()*(0.001) 
                        tvc_behavior = TaskVelocityControl(list(desired_position), robot, sim_params.dt, 3, vel_limits=vel_limits)
                        print("from model:", desired_position)
                        print("ground truth: ", goal_position)   
                    else:
                        pos, rot_mat = model(current_pc_tensor, goal_pc_tensor) 
                        pos *= 0.001
                        pos, rot_mat = pos.detach().cpu().numpy(), rot_mat.detach().cpu().numpy()
                        
                        # if max(abs(pos.squeeze())) >= 0.08:
                        #     pos *= (0.05/max(abs(pos.squeeze())))

                        pos *= 3/4

                        if first_time and h <= 0.25:
                            # pos[0][2] = 0.05
                            pos[0][2] = max(0.05, pos[0][2])
                            temp = max(w / 2 * 0.5, abs(pos[0][0]))
                            pos[0][0] *= temp/abs(pos[0][0])
                            first_time = False

                        # if not ()
                        


                        desired_pos = (pos + init_pose[:3,3]).flatten()
                        desired_rot = rot_mat @ init_pose[:3,:3]



                        tvc_behavior = TaskVelocityControl2([*desired_pos, desired_rot], robot, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                            pos_threshold = 2e-3, ori_threshold=5e-2)

                        temp1 = np.eye(4)
                        temp1[:3,:3] = rot_mat
                        temp2 = np.eye(4)
                        temp2[:3,:3] = goal_rot            
                        print("pos, rot_mat:", pos, transformations.euler_from_matrix(temp1))
                        print("goal_pos, goal_rot:", goal_pos, transformations.euler_from_matrix(temp2)) 

            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))

            state = "move to goal"


        if state == "move to goal":           
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
                rospy.logerr("Lost contact with robot")
                state = "reset" 
                # final_node_distances.append(999) 
                
                chamfer_dist = np.linalg.norm(full_pc_goal - get_point_cloud())                
                saved_nodes.append(chamfer_dist)
                final_node_distances.append(999+min(saved_nodes)) 

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(down_sampling(get_partial_point_cloud(i)))   
                chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))                
                saved_chamfers.append(chamfer_dist)
                final_chamfer_distances.append(999+min(saved_chamfers))         

                print("***final chamfer distance: ", min(saved_nodes))

                goal_count += 1
            
            else:
                if timeit.default_timer() - shapesrv_start_time >= max_shapesrv_time:
                    rospy.logerr("Timeout")
                    state = "reset" 
                    
                    chamfer_dist = np.linalg.norm(full_pc_goal - get_point_cloud())                    
                    saved_nodes.append(chamfer_dist)
                    final_node_distances.append(1999+min(saved_nodes)) 

                    current_pc = down_sampling(get_partial_point_cloud(i))
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(current_pc)   
                    chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))                    
                    saved_chamfers.append(chamfer_dist)
                    final_chamfer_distances.append(1999+min(saved_chamfers)) 

                    print("***final chamfer distance: ", min(saved_nodes))
                    
                    goal_count += 1

                else:
                    action = tvc_behavior.get_action()  
                    if action is None or gym.get_sim_time(sim) - closed_loop_start_time >= 3:#1.5:   
                        _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                        init_eulers = transformations.euler_from_matrix(init_pose) 
                        state = "get shape servo plan"    
                    else:
                        # print("desired vel:", action.get_joint_position())
                        gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())     
                        # _,final_pose = get_pykdl_client(robot.get_arm_joint_positions())   
                        # rospy.logerr(f"true delta: {final_pose[:3,3]-init_pose[:3,3]}")

                    # Terminal conditions
                    if no_rot:
                        converge = all(abs(desired_position) <= 0.005)
                    else:
                        converge = all(abs(pos.squeeze()) <= 0.005)
                    if converge or chamfer_dist < min_chamfer_dist:
                        
                        node_dist = np.linalg.norm(full_pc_goal - get_point_cloud())             
                        print("***final chamfer distance: ", node_dist)
                        final_node_distances.append(node_dist) 


                        current_pc = down_sampling(get_partial_point_cloud(i))
                        pcd = open3d.geometry.PointCloud()
                        pcd.points = open3d.utility.Vector3dVector(current_pc)  
                        chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                        final_chamfer_distances.append(chamfer_dist) 

                        goal_count += 1

                        state = "reset" 



        if state == "reset":   
            rospy.loginfo("**Current state: " + state)
            frame_count = 0
            saved_chamfers = []
            saved_nodes = []
            
            rospy.logwarn(("=== JUST ENDED goal_count " + str(goal_count)))

            
            # Go to next goal pc
            if goal_count < max_goal_count:
                goal_pc_numpy = down_sampling(goal_datas[goal_count]["partial pcs"][1])
                goal_pc_tensor = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)
                pcd_goal = open3d.geometry.PointCloud()
                pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
                pcd_goal.paint_uniform_color([1,0,0])

                full_pc_goal = goal_datas[goal_count]["full pcs"][1]

                if no_rot:
                    goal_position = goal_datas[goal_count]["pos"].T.squeeze()
                else:
                    goal_pos = goal_datas[goal_count]["pos"] 
                    goal_rot = goal_datas[goal_count]["rot"]         
                saved_obj_contact_state = goal_datas[goal_count]["obj contact state"]
                saved_robot_contact_state = goal_datas[goal_count]["robot contact state"]

            # gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], saved_robot_contact_state, gymapi.STATE_ALL) 
            # gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_obj_contact_state))
            # print("Sucessfully reset robot and object")
            # state = "get shape servo plan"

            shapesrv_start_time = timeit.default_timer()
            
            state = "home"
            first_time = True

            if fail_mtp:
                state = "home"  
                fail_mtp = False

            # final_data = {"node": final_node_distances, "chamfer": final_chamfer_distances, "num_nodes": full_pc_goal.shape[0]}
            # with open(os.path.join(chamfer_recording_path, args.obj_name + ".pickle"), 'wb') as handle:
            #     pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


        if  goal_count >= max_goal_count:                    
            all_done = True 
            # # final_data = final_node_distances
            # final_data = {"node": final_node_distances, "chamfer": final_chamfer_distances}
            # with open(os.path.join(chamfer_recording_path, args.obj_name + ".pickle"), 'wb') as handle:
            #     pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


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
    # print("total data pt count: ", data_point_count)
