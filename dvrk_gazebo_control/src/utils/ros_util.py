import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
import rospy


def interpolate_joint_trajectory(nominal_traj, dt, duration=3):
    # duration = nominal_traj.points[-1].time_from_start.to_sec()
    old_time = np.linspace(0, duration, len(nominal_traj.points))
    new_time = np.linspace(0, duration, int(duration / float(dt)))
    old_pos = np.stack([p.positions for p in nominal_traj.points])
    old_vel = np.stack([p.velocities for p in nominal_traj.points])
    old_acc = np.stack([p.accelerations for p in nominal_traj.points])
    n_dims = old_pos.shape[-1]
    new_pos = np.stack([np.interp(new_time, old_time, old_pos[:,i]) for i in range(n_dims)], axis=-1)
    new_vel = np.stack([np.interp(new_time, old_time, old_vel[:,i]) for i in range(n_dims)], axis=-1)
    new_acc = np.stack([np.interp(new_time, old_time, old_acc[:,i]) for i in range(n_dims)], axis=-1)

    new_traj = JointTrajectory()
    new_traj.joint_names = nominal_traj.joint_names
    for t in range(len(new_time)):
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(new_time[t])
        point.positions = new_pos[t,:]
        point.velocities = new_vel[t,:]
        point.accelerations = new_acc[t,:]
        new_traj.points.append(point)
    return new_traj

def convert_list_to_Pose(pose):
    """
    convert a 7-dimension pose vector (position + orientation) to ROS Pose() message type
    """
    if len(pose) != 7:
        raise ValueError(f"Expected target pose to be length 7 but got {pose}")
    converted_pose = Pose()
    converted_pose.position.x = pose[0]
    converted_pose.position.y = pose[1]
    converted_pose.position.z = pose[2]
    converted_pose.orientation.x = pose[3]
    converted_pose.orientation.y = pose[4]
    converted_pose.orientation.z = pose[5]
    converted_pose.orientation.w =  pose[6]   
    return converted_pose    
    
      