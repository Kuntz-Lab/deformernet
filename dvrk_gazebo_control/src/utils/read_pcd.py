import open3d
import os

os.chdir('/home/baothach/dvrk_grasp_data/visual/pcd') 


print("Testing IO for point cloud ...")
pcd = open3d.io.read_point_cloud("object_1_advil_liqui_gels_grasp_0.pcd")
open3d.visualization.draw_geometries([pcd])