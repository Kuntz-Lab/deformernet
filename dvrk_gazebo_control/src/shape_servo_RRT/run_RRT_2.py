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
from copy import copy
import rospy

from geometry_msgs.msg import PoseStamped, Pose
from GraspDataCollectionClient import GraspDataCollectionClient
import open3d

from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from utils.isaac_utils import fix_object_frame
import pickle

import timeit
from copy import deepcopy

# import torch
from robotRRT import RobotRRT, pcd_ize, down_sampling
from PIL import Image

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl2





ROBOT_Z_OFFSET = 0.20
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 0.86



def init():
    for i in range(num_envs):

        # # Kuka 2
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][4] = 0.24
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        gym.set_actor_dof_states(envs[i], kuka_handles_2[i], davinci_dof_states, gymapi.STATE_POS)

def get_point_cloud():
    gym.refresh_particle_state_tensor(sim)
    particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
    point_cloud = particle_state_tensor.numpy()[:, :3]  

    return point_cloud.astype('float32')

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
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"},
            {"name": "--goal_idx", "type": int, "default": 0, "help": "which goal to evaluate?"}])

    num_envs = args.num_envs
    args.headless = args.headless == "True"



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

    # load robot assets
    pose_2 = gymapi.Transform()
    pose_2.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    pose_2.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.001


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



    
    # Soft object asset
    # asset_root = "/home/baothach/sim_data/Custom/Custom_urdf"
    # soft_asset_file = "box.urdf"
    asset_root = "/home/baothach/shape_servo_data/evaluation/urdf/box_1k/inside"
    soft_asset_file = "box_0.urdf"
    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.0, -0.42, 0.04/2*0.5)
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.0005 #0.001

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
    cam_positions.append(gymapi.Vec3(0.17, -0.62, 0.2))
    cam_targets.append(gymapi.Vec3(0.0, 0.40-two_robot_offset, 0.01))
    
    for i, env_obj in enumerate(envs_obj):
        # for c in range(len(cam_positions)):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])



    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

        

    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')
    rospy.logerr(f"Running goal_idx {args.goal_idx} ... ")

  


    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
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
    dc_client = GraspDataCollectionClient()
    max_run_FEM_time = 5
    num_image = 0
    start_vis_cam = False
    prepare_vis_cam = False
    
    
    goal_data_path = "/home/baothach/shape_servo_data/comparative_study/goal_data"
    with open(os.path.join(goal_data_path, f"sample {args.goal_idx}.pickle"), 'rb') as handle:
        goal_pc = pickle.load(handle)["partial pc"]
  
    ### Set up RRT
    lims = np.array([[-1.605, 1.5994], [-0.93556, 0.94249], [-0.94249, 0.93556], [-0.93556,0.94249],
                    [0.042, 0.24], [-3.14, 3.14], [-1.5708, 1.5708], [-1.5708, 1.5708]])
    robot_rrt = RobotRRT(num_samples=10000, num_dimensions=8, step_length = 1, lims=lims, goal_pc=goal_pc, success_threshold=0.1)  

    result_path = "/home/baothach/shape_servo_data/comparative_study/RRT/results"
    os.makedirs(result_path, exist_ok=True)
    robot = Robot(gym, sim, envs[0], kuka_handles_2[0])
    all_chamfers = []   # chamfer dist at every new leaf node
    all_times = []   # computation time up until a new leaf node
    all_pcs = []    # point clouds
    all_full_pcs = []
    total_pc_time = 0
    
    start_time = timeit.default_timer()    

    close_viewer = False


    while (not close_viewer) and (not all_done): 
       
        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if state == "home" :   
            frame_count += 1
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.24)            
            if frame_count == 10:
                rospy.loginfo("**Current state: " + state + ", current sample count: " + str(sample_count))
                

                if first_time:                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    saved_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
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
                    
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), 0.35)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), -0.35)         
                       
                gym.refresh_particle_state_tensor(sim)
                init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                init_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))  
                init_joint_states = deepcopy(gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos'][:8])
                ee_pos = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3]["pose"]["p"])
                
                robot_rrt.save_init_states(init_object_state, init_robot_state, init_joint_states, ee_pos)                
                
                start_RRT_time = timeit.default_timer()
                state = "run RRT"


        if state == "run RRT":

            if frame_count % 5 == 0:
                main_ins_pos = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"))
                if main_ins_pos <= 0.042:
                    rospy.logerr("Exceed joint constraint")
                    state = "reset"

                # contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
                # if not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
                #     print("Lost contact with robot")
                #     state = "reset"
                    
            frame_count += 1

            # print("**RUNNING RRT")
            if robot_rrt.run_FEM_is_complete:
                if robot_rrt.invalid_state_occur == False:
                    gym.refresh_particle_state_tensor(sim)
                    object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))                           
                    robot_rrt.update_node_object_state(object_state, robot_state)

                    start_pc_time = timeit.default_timer()
                    pc = down_sampling(get_partial_point_cloud(i), num_pts=1024)
                    # pc = down_sampling(get_point_cloud(), num_pts=1024)
                    pcd = pcd_ize(pc)
                    pcd.paint_uniform_color([0,0,0])
                    chamf_dist = robot_rrt.compute_chamfer_dist(pcd, vis=False)  
                    rospy.logerr(f"chamf_dist: {chamf_dist}")   
                    
                    total_pc_time += timeit.default_timer() - start_pc_time
                    print("total_pc_time:", total_pc_time)
                    print("total computation time:", timeit.default_timer() - start_RRT_time)
                    RRT_time = timeit.default_timer() - total_pc_time - start_RRT_time
                    rospy.logwarn(f"RRT_time: {RRT_time}")
                    
                    all_chamfers.append(deepcopy(chamf_dist))
                    all_times.append(deepcopy(RRT_time))
                    # all_pcs.append(down_sampling(pc, num_pts=256))
                    all_pcs.append(pc)
                    all_full_pcs.append(get_point_cloud())

                    if robot_rrt.is_goal(current_pc=pc, goal_oriented=True, threshold = robot_rrt.success_threshold) \
                    or robot_rrt.reach_max_samples:
                        robot_rrt.found_path = True
                        robot_rrt.reach_max_samples = True  # Final chamfer < threshold --> success
                        rospy.logerr(f"==== END !!! ====")
                        
                    else:                
                        new_node, parent_node = robot_rrt.build_rrt()
                        print("robot_rrt.sample_count:", robot_rrt.sample_count)
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
                    rospy.logerr("====Running FEM timeout====")
                robot_rrt.T.nodes[-1].state = deepcopy(gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos'][:8])                


        if robot_rrt.reach_max_samples:
            rospy.logerr("RRT reached max sample")         
            
            best_chamfer_idx = np.argmin(all_chamfers)  
            rospy.logerr(f"Best chamfer: {all_chamfers[best_chamfer_idx]}")  
            rospy.logerr(f"Best time: {all_times[best_chamfer_idx]}") 
            # best_pcd = pcd_ize(all_pcs[best_chamfer_idx])
            # best_pcd.paint_uniform_color([0,0,0])
            # open3d.visualization.draw_geometries([best_pcd, robot_rrt.pcd_goal])

            result_data = {"chamfer": all_chamfers, "time": all_times, "pc": all_pcs, "full pc": all_full_pcs}
            with open(os.path.join(result_path, f"sample {args.goal_idx}.pickle"), 'wb') as handle:
                pickle.dump(result_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            all_done = True

        if state == "reset":   
            rospy.loginfo("**Current state: " + state)
            
            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], saved_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, [0]*8)
            
            print("Sucessfully reset robot and object")
           
            frame_count = 0
            all_chamfers = []
            all_times = []
            all_pcs = []
            all_full_pcs = []
            total_pc_time = 0
            
                

            gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)  
            gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, [0,0,0,0,0.24,0,0,0,1.5,0.8]) 


            state = "home"


        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)

    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

