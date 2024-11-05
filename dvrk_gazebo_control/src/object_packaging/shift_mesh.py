import numpy as np
import trimesh
import trimesh.visual

# Load the .obj file
mesh = trimesh.load_mesh('/home/baothach/sim_data/Custom/Custom_mesh/object_packaging/amazon_box.obj')

# Extract vertices from the loaded mesh
vertices = mesh.vertices

# Find the minimum y-value (the bottom of the box)
min_y_value = np.min(vertices[:, 1])

# Get vertices that are at the bottom (with min_y_value)
bottom_vertices = vertices[vertices[:, 1] == min_y_value]

# Now, find the extremes in x and z directions (min and max x, min and max z)
min_x = np.min(bottom_vertices[:, 0])
max_x = np.max(bottom_vertices[:, 0])
min_z = np.min(bottom_vertices[:, 2])
max_z = np.max(bottom_vertices[:, 2])

# These four points define the corner vertices of the bottom
corner_vertices = np.array([
    [min_x, min_y_value, min_z],
    [min_x, min_y_value, max_z],
    [max_x, min_y_value, min_z],
    [max_x, min_y_value, max_z]
])

# # Calculate the midpoint of the four corner vertices
# midpoint = np.mean(corner_vertices, axis=0)

# # Shift all vertices so that the midpoint is the new origin
# shifted_vertices = vertices - midpoint

# # Update the mesh with the shifted vertices
# mesh.vertices = shifted_vertices

# Add a coordinate frame for visualization
coordinate_frame = trimesh.creation.axis(origin_size=0.02)  # Optional size parameter for the axis
scene = trimesh.Scene([mesh, coordinate_frame])

# Visualize the scene with the mesh and the coordinate frame
scene.show()

# # Optionally, save the shifted mesh
# output_file_path = '/home/baothach/sim_data/Custom/Custom_mesh/object_packaging/amazon_box_shifted.obj'
# mesh.export(output_file_path)
# print(f"Shifted mesh saved to {output_file_path}")
