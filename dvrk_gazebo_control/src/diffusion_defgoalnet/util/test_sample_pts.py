import numpy as np
import open3d as o3d

def sample_cylindrical_points(num_points, r_min, r_max, theta_min, theta_max, z_min, z_max):
    points = []
    for _ in range(num_points):
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(theta_min, theta_max)
        z = np.random.uniform(z_min, z_max)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    
    return np.array(points)

# Parameters
num_points = 1000
r_min, r_max = 1.0, 5.0
theta_min, theta_max = 0, np.pi  # Half circle
z_min, z_max = -3.0, 3.0

# Sample points
points = sample_cylindrical_points(num_points, r_min, r_max, theta_min, theta_max, z_min, z_max)

# Create an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Origin marker (small red sphere at (0, 0, 0))
origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
origin_sphere.paint_uniform_color([1, 0, 0])  # Red color
origin_sphere.translate([0, 0, 0])

# Coordinate axes for reference
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

# Visualization
o3d.visualization.draw_geometries([pcd, origin_sphere, axes])
