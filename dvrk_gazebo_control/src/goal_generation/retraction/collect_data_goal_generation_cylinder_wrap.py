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
from copy import deepcopy
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from GraspDataCollectionClient import GraspDataCollectionClient
import open3d
from utils import open3d_ros_helper as orh
from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from utils.isaac_utils import fix_object_frame, get_pykdl_client
import pickle
import timeit
from copy import deepcopy
from core import Robot
from behaviors import MoveToPose
from cylinder_wrap_utils import *



ROBOT_Z_OFFSET = 0.2
two_robot_offset = 0.86


def init():
    for i in range(num_envs):
        davinci_dof_states = gym.get_actor_dof_states(envs[i], dvrk_handles[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        davinci_dof_states['pos'][4] = 0.24
        gym.set_actor_dof_states(envs[i], dvrk_handles[i], davinci_dof_states, gymapi.STATE_POS)

def get_point_cloud():
    """
    Get full point cloud of the deformable object
    """
    
    gym.refresh_particle_state_tensor(sim)
    particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
    point_cloud = particle_state_tensor.numpy()[:, :3]  
    
    return point_cloud.astype('float32')


def get_partial_point_cloud(obj_name="deform_tube"):

    """
    Get partial point cloud of any object
    """

    # if obj_name == "deform_tube":
    #     cam_handles = deform_tube_cam_handles
    #     cam_width = deform_tube_cam_width
    #     cam_height = deform_tube_cam_height
    # elif obj_name == "rigid_cylinder":
    #     cam_handles = rigid_cylinder_cam_handles
    #     cam_width = rigid_cylinder_cam_width
    #     cam_height = rigid_cylinder_cam_height

    # Render all of the image sensors only when we need their output here
    # rather than every frame.
    gym.render_all_camera_sensors(sim)

    points = []
    print("Converting Depth images to point clouds. Have patience...")
    
    depth_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_SEGMENTATION)

    # Get the camera view matrix and invert it to transform points from camera to world space   
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs_obj[i], cam_handles[0])))

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    # Ignore any points that don't belong to the object
    if obj_name == "deform_tube":
        depth_buffer[seg_buffer == 11] = -10001
        depth_buffer[seg_buffer == 10] = -10001
    elif obj_name == "rigid_cylinder":   
        depth_buffer[seg_buffer != 10] = -10001 

    # Compute point cloud
    centerU = cam_width/2
    centerV = cam_height/2
    for k in range(cam_width):
        for t in range(cam_height):
            if obj_name == "deform_tube":
                if depth_buffer[t, k] < -3:
                    continue
            elif obj_name == "rigid_cylinder":
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

    return np.array(points).astype('float32')


if __name__ == "__main__":

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="dvrk Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--obj_name", "type": str, "default": 'cylinder_0', "help": "select variations of a primitive shape"},
            {"name": "--headless", "type": bool, "default": False, "help": "headless mode"},
            {"name": "--data_category", "type": str, "default": "deformernet", "help": "deformernet or MP"}])

    num_envs = args.num_envs
    


    # configure sim
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    if sim_type is gymapi.SIM_FLEX:
        sim_params.substeps = 4
        sim_params.dt = 1./60.
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4#4
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

    # load robot asset
    dvrk_pose = gymapi.Transform()
    dvrk_pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    dvrk_asset_options = gymapi.AssetOptions()
    dvrk_asset_options.armature = 0.001
    dvrk_asset_options.fix_base_link = True
    dvrk_asset_options.thickness = 0.001
    dvrk_asset_options.fix_base_link = True
    dvrk_asset_options.flip_visual_attachments = False
    dvrk_asset_options.collapse_fixed_joints = True
    dvrk_asset_options.disable_gravity = True
    dvrk_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    asset_root = "./src/dvrk_env"
    dvrk_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"
    print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
    dvrk_asset = gym.load_asset(sim, asset_root, dvrk_asset_file, dvrk_asset_options)

    if sim_type is gymapi.SIM_FLEX:
        dvrk_asset_options.max_angular_velocity = 40000.

    
    # Load soft tube asset
    soft_asset_root = '/home/baothach/sim_data/Custom/Custom_urdf/multi_cylinders/'
    soft_asset_file = 'cylinder_2_attached.urdf'

    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0, 0.4-two_robot_offset, 0.02/2.0)
    soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
    soft_thickness = 0.001#0.0005    # important to add some thickness to the soft body to avoid interpenetrations

    soft_asset_options = gymapi.AssetOptions()
    soft_asset_options.fix_base_link = True
    soft_asset_options.thickness = soft_thickness
    soft_asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, soft_asset_root, soft_asset_file, soft_asset_options)


    # Load rigid cylinder asset
    cylinder_asset_options = gymapi.AssetOptions()
    cylinder_asset_options.fix_base_link = True
    cylinder_asset_options.thickness = 0.003
    cylinder_asset_options.disable_gravity = True
    cylinder_asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/multi_cylinders"
    cylinder_asset_file = "cylinder_wrap_tissue.urdf"   #trimesh.creation.cylinder(radius=0.015, height=0.1)      
    cylinder_pose = gymapi.Transform()
    cylinder_pose.p = gymapi.Vec3(-0.02, 0.40001-two_robot_offset, 0.04)
    cylinder_asset = gym.load_asset(sim, cylinder_asset_root, cylinder_asset_file, cylinder_asset_options)        
    
   
    # set up the env grid
    spacing = 0.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
  

    # cache some common handles for later use
    envs = []
    envs_obj = []
    dvrk_handles = []
    object_handles = []
    

    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    base_poses = []

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add dvrk
        dvrk_handle = gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i, 1, segmentationId=11)  
        dvrk_handles.append(dvrk_handle)      
        
        # add soft obj        
        env_obj = env
        envs_obj.append(env_obj)                
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)
        object_handles.append(soft_actor)

        # add rigid cylinder
        cylinder_actor = gym.create_actor(env, cylinder_asset, cylinder_pose, "cylinder", i, 0, segmentationId=10)
        color = gymapi.Vec3(1,0,0)
        gym.set_rigid_body_color(env, cylinder_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        
    dvrk_dof_props = gym.get_asset_dof_properties(dvrk_asset)
    dvrk_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dvrk_dof_props["stiffness"].fill(200.0)
    dvrk_dof_props["damping"].fill(40.0)
    dvrk_dof_props["stiffness"][8:].fill(1)
    dvrk_dof_props["damping"][8:].fill(2)  
    vel_limits = dvrk_dof_props['velocity']  
    

    # Camera for viewer setup
    if not args.headless:
        cam_pos = gymapi.Vec3(0.3, -0.7, 0.3)
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
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])


    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, dvrk_handles[i], dvrk_dof_props)


    '''
    Main stuff is here
    '''
    rospy.init_node('cylinder_wrap')
 

    #### Some important paramters
    init()  # Initilize robot joints
    all_done = False
    state = "home"
    start_time = timeit.default_timer()    
    close_viewer = False
        
    data_recording_path = "/home/baothach/shape_servo_data/goal_generation/cylinder_wrap/data"
    os.makedirs(data_recording_path, exist_ok=True)

    sample_count = 0
    frame_count = 0
    max_group_count = 1500000

    pc_on_trajectory = []
    full_pc_on_trajectory = []
    first_time = True

    dc_client = GraspDataCollectionClient()
    robot = Robot(gym, sim, envs[0], dvrk_handles[0])

    
    #### Cylinder wrap specific stuff
    visualization = True
    num_samples, num_circle, z_shift = 10, 1, 0.02
    center, radius, slope = [-0.02,0.40001-two_robot_offset], 0.03, 0.006      
    # Get spiral checkpoints
    goal_spiral = generate_spiral(center, radius, slope, num_circle=num_circle, z_shift=z_shift, num_samples=num_samples)
    
    group_count = int(len(os.listdir(data_recording_path)) / num_samples)   
       

    while (not close_viewer) and (not all_done):   

        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 

        if state == "home" :   
            frame_count += 1
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "dvrk", "psm_main_insertion_joint"), 0.24)            
            if frame_count == 5:
                rospy.loginfo("**Current state: " + state + ", current sample count: " + str(sample_count))
                

                if first_time:   
                    
                    # Save robot and object states for reset                  
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], dvrk_handles[i], gymapi.STATE_ALL))
                    first_time = False   
                                   
                    # Get current object state
                    current_particle_state = get_point_cloud()
                    pcd = pcd_ize(current_particle_state)
                    open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                    pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                    pc_ros_msg = fix_object_frame(pc_ros_msg)
                    
                    # Get rigid cylinder pc              
                    cylinder_pc = get_partial_point_cloud(obj_name="rigid_cylinder")                    
         
                    if visualization:
                        pcd = pcd_ize(get_partial_point_cloud(obj_name="deform_tube"), color=[0,0,0])
                        pcd_cylinder = pcd_ize(cylinder_pc, color=[1,0,0])
                        pcd_goal = pcd_ize(goal_spiral, color=[0,0,1])
                        open3d.visualization.draw_geometries([pcd, pcd_cylinder, pcd_goal])       
                
                spiral_idx = 0
                frame_count = 0
                state = "generate preshape"         


        ############################################################################
        # generate preshape: Get a manipulation point
        ############################################################################
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


        ############################################################################
        # move to preshape: Move robot gripper to the manipulation point location
        ############################################################################
        if state == "move to preshape":         
            action = mtp_behavior.get_action()

            if action is not None:
                gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())      
                        
            if mtp_behavior.is_complete():
                state = "grasp object"   
                rospy.loginfo("Succesfully executed PRESHAPE moveit arm plan. Let's grasp it!!") 


        ############################################################################
        # grasp object: close gripper
        ############################################################################              
        if state == "grasp object":             
            rospy.loginfo("**Current state: " + state)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper1_joint"), -2.5)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper2_joint"), -3.0)         

            g_1_pos = 0.35
            g_2_pos = -0.35
            dof_states = gym.get_actor_dof_states(envs[i], dvrk_handles[i], gymapi.STATE_POS)
            if dof_states['pos'][8] < 0.35:
                                       
                state = "get shape servo plan"
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper1_joint"), g_1_pos)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "dvrk", "psm_tool_gripper2_joint"), g_2_pos) 


        ############################################################################
        # get shape servo plan: Set up MoveIt based on the spiral checkpoints
        ############################################################################
        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 

            # Set up eef goal pose, based on the spiral checkpoints
            cartesian_pose = Pose()
            cartesian_pose.orientation.x = 0
            cartesian_pose.orientation.y = 0.707107
            cartesian_pose.orientation.z = 0.707107
            cartesian_pose.orientation.w = 0
            cartesian_pose.position.x = -goal_spiral[spiral_idx][0]
            cartesian_pose.position.y = -goal_spiral[spiral_idx][1]
            cartesian_pose.position.z = goal_spiral[spiral_idx][2] - ROBOT_Z_OFFSET

            # Set up moveit
            plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=robot.get_full_joint_positions())
            if (not plan_traj):
                rospy.logerr('Can not find moveit plan to shape servo. Ignore this grasp.\n')  
                state = "reset"
            else:
                state = "move to goal"
                traj_index = 0

        ############################################################################
        # move to goal: Move robot end-effector to the desired pose using MoveIt
        ############################################################################            
        if state == "move to goal":
            
            # robot exeeds joint constraint(s)
            main_ins_pos = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "dvrk", "psm_main_insertion_joint"))
            if main_ins_pos <= 0.042:
                rospy.logerr("Exceed joint constraint")
                state = "reset"

            # soft tube lost contact w robot
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if not(9 in contacts or 10 in contacts):  
                rospy.logerr("Lost contact with robot")
                state = "reset"

            else:
                if frame_count == 0:
                    full_pc_on_trajectory.append(get_point_cloud())
                    pc_on_trajectory.append(get_partial_point_cloud(obj_name="deform_tube"))                       
                frame_count += 1
                
                dof_states = robot.get_full_joint_positions()
                plan_traj_with_gripper = [plan+[g_1_pos,g_2_pos] for plan in plan_traj]
                pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
                gym.set_actor_dof_position_targets(envs[0], dvrk_handles[0], pos_targets)                
                
                # Set target joint positions
                if traj_index <= len(plan_traj) - 2:
                    if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.1):
                        traj_index += 1 
                else:
                    if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.01):
                        traj_index += 1 

                # Complete 1 checkpoint
                if traj_index == len(plan_traj):
                    rospy.logerr(f"Successful checkpoint {spiral_idx}!")
                    traj_index = 0
                    spiral_idx += 1

                    if visualization:
                        pcd = pcd_ize(get_partial_point_cloud(obj_name="deform_tube"), color=[0,0,0])
                        open3d.visualization.draw_geometries([pcd, pcd_cylinder, pcd_goal])

                    # Collect full and partial point clouds along the trajectory (every time robot completes 1 checkpoint)
                    if spiral_idx < num_samples:
                        full_pc_on_trajectory.append(get_point_cloud())
                        pc_on_trajectory.append(get_partial_point_cloud(obj_name="deform_tube"))
                        state = "get shape servo plan"      
                    else:
                        pc_goal = get_partial_point_cloud(obj_name="deform_tube")
                        full_pc_goal = get_point_cloud()

                        for j in range(len(pc_on_trajectory)):                                                    
                            # Form input-output pair for training DNN
                            partial_pcs = (pc_on_trajectory[j], pc_goal)
                            full_pcs = (full_pc_on_trajectory[j], full_pc_goal)

                            data = {"full pcs": full_pcs, "partial pcs": partial_pcs,\
                                    "cylinder_pc": cylinder_pc}
                            with open(os.path.join(data_recording_path, f"group {group_count} sample {sample_count}.pickle"), 'wb') as handle:
                                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                                
                            sample_count += 1
                        
                        rospy.logerr(f"DONE traj {group_count}")                                  
                        group_count += 1
                        state = "reset"     


        ############################################################################
        # Reset: Reset robot and object to the initial state
        ############################################################################  
        if state == "reset":   
            rospy.loginfo("**Current state: " + state)
            
            frame_count = 0
            sample_count = 0
            pc_on_trajectory = []
            full_pc_on_trajectory = []
            # curr_trans_on_trajectory = []           

            gym.set_actor_rigid_body_states(envs[i], dvrk_handles[i], init_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, [0]*8)            
            print("Sucessfully reset robot and object")

            state = "home"
 
        if group_count >= max_group_count:                    
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

