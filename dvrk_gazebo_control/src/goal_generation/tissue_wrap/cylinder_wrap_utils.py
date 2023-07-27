import open3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

def pcd_ize(pc, color=None, vis=False):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)   
    if color is not None:
        pcd.paint_uniform_color(color) 
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd

def generate_spiral(center, radius, slope, num_circle=1, z_shift=0, num_samples=10):

    # theta = np.linspace(0, 2*np.pi, num_samples)
    theta = np.linspace(-np.pi, np.pi, num_samples)
    theta = np.tile(theta, num_circle)  # duplicate circle multiple times
    x = center[0] + radius*np.cos(theta)
    y = center[1] + radius*np.sin(theta)
    z = np.array([i for i in range(0, num_samples*num_circle)])*slope + z_shift
    
    return np.column_stack((x,y,z))

# center, radius, slope = [0,0], 0.1, 0.001
# spiral = generate_spiral(center, radius, slope, num_circle=2, z_shift=0, num_samples=100)
# print(spiral.shape)
# pcd = pcd_ize(spiral)
# open3d.visualization.draw_geometries([pcd])

