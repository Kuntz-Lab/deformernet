import numpy as np
import trimesh

# Load the .obj file and extract vertices
obj_file_path = '/home/baothach/sim_data/Custom/Custom_mesh/object_packaging/amazon_box.obj'

vertices = []
with open(obj_file_path, 'r') as file:
    for line in file:
        if line.startswith('v '):
            vertex = list(map(float, line.strip().split()[1:4]))
            vertices.append(vertex)

# Convert to numpy array for easier manipulation
vertices = np.array(vertices)

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

# Define edges as pairs of corner vertices
def edge_length(v1, v2):
    return np.linalg.norm(v1 - v2)
edges = [
    (corner_vertices[0], corner_vertices[1]),  # Edge between (min_x, min_z) and (min_x, max_z)
    (corner_vertices[1], corner_vertices[3]),  # Edge between (min_x, max_z) and (max_x, max_z)
    (corner_vertices[3], corner_vertices[2]),  # Edge between (max_x, max_z) and (max_x, min_z)
    (corner_vertices[2], corner_vertices[0])   # Edge between (max_x, min_z) and (min_x, min_z)
]

# Calculate and print edge lengths
for i, (v1, v2) in enumerate(edges):
    length = edge_length(v1, v2)
    print(f"Edge {i+1} length: {length}")

# # Load the mesh using trimesh
# mesh = trimesh.load(obj_file_path)

# # Create spheres at the correct corner vertices
# spheres = [trimesh.primitives.Sphere(radius=0.005, center=vertex) for vertex in corner_vertices]

# # Combine the spheres with the original mesh
# scene = trimesh.Scene([mesh] + spheres)


# # Show the combined mesh and spheres
# scene.show()
