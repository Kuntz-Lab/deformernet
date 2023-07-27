import pickle
import open3d 
import numpy as np
import os
import matplotlib.pyplot as plt
from util_functions import *

def normalize_list(lst):
    minimum = min(lst)
    maximum = max(lst)
    value_range = maximum - minimum

    normalized_lst = [(value - minimum) / value_range for value in lst]

    return normalized_lst

def compute_von_mises_stress(cauchy_tensor):

    """
    To compute the Von Mises stress from the Cauchy stress tensor, you can follow these steps:

    1. Compute the deviatoric stress tensor, denoted as S:

    - Subtract the mean stress component from the Cauchy stress tensor to obtain the deviatoric stress tensor.
    - The mean stress component is calculated as the average of the three principal stresses: σ_mean = (σ_xx + σ_yy + σ_zz) / 3.
    - The deviatoric stress tensor S is obtained by subtracting the mean stress component from the Cauchy stress tensor:
        S = σ - σ_mean * I, where σ is the Cauchy stress tensor, and I is the identity tensor.
    
    2. Compute the Von Mises stress (σ_VM) from the deviatoric stress tensor S:
    - Compute the second invariant of the deviatoric stress tensor, J2:
        J2 = (1/2) * (S_ij * S_ij), where S_ij represents the components of the deviatoric stress tensor.
    - Calculate the Von Mises stress as the square root of three times the second invariant:
        σ_VM = sqrt(3 * J2).    
    
    """

    # Compute the deviatoric stress tensor
    mean_stress = np.mean(cauchy_tensor)
    deviatoric_stress = cauchy_tensor - mean_stress * np.identity(3)

    # Compute the second invariant of the deviatoric stress tensor
    deviatoric_stress_squared = np.square(deviatoric_stress)
    deviatoric_stress_sum = np.sum(deviatoric_stress_squared)
    j2 = 0.5 * deviatoric_stress_sum

    # Compute the Von Mises stress
    von_mises_stress = np.sqrt(3 * j2)

    return von_mises_stress


def compute_stress_each_vertex(tet_stress, adjacent_tetrahedral_dict, vertex_idx):
    """
    The stress tensor at each vertex is calculated by averaging
    the stress tensors at all adjacent tetrahedral elements. Each stress tensor is
    then converted to the scalar von Mises stress (i.e., the second
    invariant of the deviatoric stress), which is widely used to
    quantify whether a material has yielded.     
    
    """
    avg_cauchy_stress_tensor = np.zeros((3,3))
    for tetrahedra_idx in adjacent_tetrahedral_dict[vertex_idx]:    

        cauchy_stress = tet_stress[tetrahedra_idx]
        cauchy_stress_matrix = np.array([[cauchy_stress.x.x, cauchy_stress.y.x, cauchy_stress.z.x],
                                        [cauchy_stress.x.y, cauchy_stress.y.y, cauchy_stress.z.y],
                                        [cauchy_stress.x.z, cauchy_stress.y.z, cauchy_stress.z.z]])
        
        avg_cauchy_stress_tensor += cauchy_stress_matrix

    avg_cauchy_stress_tensor /= len(adjacent_tetrahedral_dict[vertex_idx])  # average out over all all adjacent tetrahedras
    
    von_mises_stress = compute_von_mises_stress(avg_cauchy_stress_tensor)
    return von_mises_stress


def get_adjacent_tetrahedrals_of_vertex(tet_indices):
    """

    For each vertex in the tetrahedral mesh, find a list of tetrahedral elements that include it.
    ex: 

    mesh = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [0, 2, 4, 6]]
    Vertex 0 belongs to tetrahedra: [[0, 1, 2, 3], [0, 2, 4, 6]]
    Vertex 1 belongs to tetrahedra: [[0, 1, 2, 3]]
    Vertex 2 belongs to tetrahedra: [[0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 4, 6]]
    Vertex 3 belongs to tetrahedra: [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
    Vertex 4 belongs to tetrahedra: [[1, 2, 3, 4], [2, 3, 4, 5], [0, 2, 4, 6]]
    Vertex 5 belongs to tetrahedra: [[2, 3, 4, 5]]
    Vertex 6 belongs to tetrahedra: [[0, 2, 4, 6]]

    """
    
    
    adjacent_tetrahedral_dict = {}

    for idx, tetrahedron in enumerate(tet_indices):
        v1, v2, v3, v4 = tetrahedron

        # Add the tetrahedron to the vertex's list
        adjacent_tetrahedral_dict.setdefault(v1, []).append(idx)    # setdefault(key, default) is a dictionary method in Python that returns the value of a specified key if it exists in the dictionary. If the key does not exist, it inserts the key with a specified default value and returns the default value.
        adjacent_tetrahedral_dict.setdefault(v2, []).append(idx)
        adjacent_tetrahedral_dict.setdefault(v3, []).append(idx)
        adjacent_tetrahedral_dict.setdefault(v4, []).append(idx)

    return adjacent_tetrahedral_dict


save_path = "/home/baothach/shape_servo_data/retraction_cutting/test"
with open(os.path.join(save_path, "test_0.pickle"), 'rb') as handle:
    data = pickle.load(handle) 

(tet_indices, tet_stress) = data["tet"]
tet_indices = np.array(tet_indices).reshape(-1,4)
(tri_indices, tri_parents, tri_normals) = data["tri"]
full_pc = data["particles"]

pcd = pcd_ize(full_pc)
# colors = np.tile(np.array([[1,0,0]]), (full_pc.shape[0], 1))

adjacent_tetrahedral_dict = get_adjacent_tetrahedrals_of_vertex(tet_indices)
print("adjacent_tetrahedral_dict:", adjacent_tetrahedral_dict[0])
print("Number of keys:", len(adjacent_tetrahedral_dict))

all_stresses = []
for i in range(full_pc.shape[0]):
    all_stresses.append(compute_stress_each_vertex(tet_stress, adjacent_tetrahedral_dict, vertex_idx=i))
# print("von_mises_stress:", von_mises_stress)

print("max min stress:", max(all_stresses), min(all_stresses))
all_stresses = np.log(all_stresses)
all_stresses = np.array([normalize_list(all_stresses)]).T



colors = np.pad(all_stresses, ((0, 0), (0, 2)), mode='constant')
print("colors.shape:", colors.shape)

pcd.colors = open3d.utility.Vector3dVector(colors)
open3d.visualization.draw_geometries([pcd])  