import open3d
import pyransac3d as pyrsc
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy

def vecalign(a, b):
    '''
    Returns the rotation matrix that can rotate the 3 dimensional b vector to
    be aligned with the a vector.

    @param a (3-dim array-like): destination vector
    @param b (3-dim array-like): vector to rotate align with a in direction

    the vectors a and b do not need to be normalized.  They can be column
    vectors or row vectors
    '''
    with np.errstate(divide='raise', under='raise', over='raise', invalid='raise'):
        a = np.asarray(a)
        b = np.asarray(b)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        cos_theta = a.dot(b) # since a and be are unit vecs, a.dot(b) = cos(theta)

        # if dot is close to -1, vectors are nearly equal
        if np.isclose(cos_theta, 1):
            # TODO: solve better than just assuming they are exactly identical
            return np.eye(3)

        # if dot is close to -1, vectors are nearly opposites
        if np.isclose(cos_theta, -1):
            # TODO: solve better than just assuming they are exactly opposite
            return -np.eye(3)

        axis = np.cross(b, a)
        sin_theta = np.linalg.norm(axis)
        axis = axis / sin_theta
        c = cos_theta
        s = sin_theta
        t = 1 - c
        x = axis[0]
        y = axis[1]
        z = axis[2]

        # angle-axis formula to create a rotation matrix
        return np.array([
            [t*x*x + c  , t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c  , t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c  ],
            ])

sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
sphere.paint_uniform_color([0,0,1])
pcd_original = open3d.io.read_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd")
pcd = copy.deepcopy(pcd_original)
# open3d.visualization.draw_geometries([pcd])
pcd_original.paint_uniform_color([0,0,0])

constrain_plane = np.array([0, 1, 0, 0.45])  




# plane = open3d.geometry.TriangleMesh.create_box(width=0.2, height=0.001, depth=0.1)
# plane.paint_uniform_color([0,0,1])
# open3d.visualization.draw_geometries([pcd, plane.translate((0-0.1,-0.4,0))])
points  = np.array([p for p in np.asarray(pcd.points) if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]])

# Find rotation point:
max_z = max([p[2] for p in points])
center = pcd.get_center().reshape(3)
dist = (np.dot(center, constrain_plane[:3]) + constrain_plane[3])/np.linalg.norm(constrain_plane[:3])
unit_normal = np.array(constrain_plane[:3])/np.linalg.norm(constrain_plane[:3])
rot_pt = center - dist*unit_normal

pcd.points = open3d.utility.Vector3dVector(np.array([p for p in np.asarray(pcd.points) if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] <= -constrain_plane[3]]))
pcd.paint_uniform_color([0,0,0])

pcd2 = open3d.geometry.PointCloud()
pcd2.points = open3d.utility.Vector3dVector(points)
pcd2.paint_uniform_color([1,0,0])



plane1 = pyrsc.Plane()
best_eq, best_inliers = plane1.fit(points, thresh=0.001, maxIteration=1000)
if best_eq[2] > 0:
    best_eq = -np.array(best_eq)
# best_eq = np.array([0.02465008, 0.0274112,  0.99932027])



r = vecalign(np.array(best_eq)[:3], constrain_plane[:3])
pcd2.rotate(R = r.T, center = rot_pt.reshape((3,1)))
# open3d.visualization.draw_geometries([pcd, pcd2])
# open3d.visualization.draw_geometries([pcd + pcd2, sphere])
open3d.io.write_point_cloud("/home/baothach/shape_servo_data/new_task/test_fit_plane/goal_1.pcd", pcd+pcd2)




import sys
sys.path.append('/home/baothach/shape_servo_DNN')
from generalization_tasks.architecture import DeformerNetManiPoint
import torch
import os
# sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *

# Set up DNN:
device = torch.device("cuda")
model = DeformerNetManiPoint(normal_channel=False).to(device)
weight_path = "/home/baothach/shape_servo_data/generalization/surgical_setup/weights/run3(mani_point_regression)"
model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 412")))  
model.eval()





pc = np.asarray(pcd_original.points).astype('float32') 
pcd_goal = open3d.io.read_point_cloud("/home/baothach/shape_servo_data/new_task/test_fit_plane/goal_1.pcd")
pc_goal = np.asarray(pcd_goal.points).astype('float32') 


# print("shape: ", pc.shape, pc_goal.shape)

# farthest_indices,_ = farthest_point_sampling(pc, 1024)
# pc_resampled = pc[farthest_indices.squeeze()]

# farthest_indices_goal,_ = farthest_point_sampling(pc_goal, 1024)
# pc_goal_resampled = pc_goal[farthest_indices_goal.squeeze()]


# pc = np.array([[pt[0], pt[1], pt[2]+0.2] for pt in pc]).astype('float32')
# pc_goal = np.array([[pt[0], pt[1], pt[2]+0.2] for pt in pc_goal]).astype('float32')
# pcd_original.points = open3d.utility.Vector3dVector(pc)
# pcd_goal.points = open3d.utility.Vector3dVector(pc_goal)



print(pc.shape, pc_goal.shape)
pc = torch.from_numpy(pc).unsqueeze(0).to(device).permute(0,2,1)
pc_goal = torch.from_numpy(pc_goal).unsqueeze(0).to(device).permute(0,2,1)

desired_mani_point = model(pc, pc_goal).squeeze().cpu().detach().numpy()*(0.001)

mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
mani_point.paint_uniform_color([0,0,1])
open3d.visualization.draw_geometries([mani_point.translate((\
                                        desired_mani_point[0], desired_mani_point[1], desired_mani_point[2])), \
                                        pcd_original,
                                        pcd_goal.translate((0.2,0,0))])   