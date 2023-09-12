import numpy as np
import trimesh
import open3d
import pickle 
import transformations

def record_eval_data(final_partial_pc, final_full_pc, tri_indices, cylinder_shift, final_percent, save_path):
    data = {"final_partial_pc": final_partial_pc, "final_full_pc": final_full_pc, "tri_indices": tri_indices,
            "cylinder_shift": cylinder_shift, "final_percent": final_percent}    
    
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=3) 
        

def create_tet_mesh(mesh_dir, intput_tri_mesh_name, output_tet_mesh_name=None, mesh_extension='.stl', 
                    coarsen=True, verbose=False, fTetWild_dir='/home/baothach/fTetWild/build'):
    
    if output_tet_mesh_name is None:
        output_tet_mesh_name = intput_tri_mesh_name
    
    # surface mesh (.stl, .obj, etc.) to volumetric mesh (.mesh)
    import os
    os.chdir(fTetWild_dir) 
    mesh_path = os.path.join(mesh_dir, intput_tri_mesh_name + mesh_extension)
    save_fTetwild_mesh_path = os.path.join(mesh_dir, output_tet_mesh_name + '.mesh')
    
    if coarsen:
        os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path + " --coarsen >/dev/null")
    else:
        os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path + " >/dev/null")

    # .mesh to .tet:
    mesh_file = open(os.path.join(mesh_dir, output_tet_mesh_name + '.mesh'), "r")
    tet_output = open(
        os.path.join(mesh_dir, output_tet_mesh_name + '.tet'), "w")

    # Parse .mesh file
    mesh_lines = list(mesh_file)
    mesh_lines = [line.strip('\n') for line in mesh_lines]
    vertices_start = mesh_lines.index('Vertices')
    num_vertices = mesh_lines[vertices_start + 1]

    vertices = mesh_lines[vertices_start + 2:vertices_start + 2
                        + int(num_vertices)]

    tetrahedra_start = mesh_lines.index('Tetrahedra')
    num_tetrahedra = mesh_lines[tetrahedra_start + 1]
    tetrahedra = mesh_lines[tetrahedra_start + 2:tetrahedra_start + 2
                            + int(num_tetrahedra)]

    if verbose:
        print("# Vertices, # Tetrahedra:", num_vertices, num_tetrahedra)

    # Write to tet output
    tet_output.write("# Tetrahedral mesh generated using\n\n")
    tet_output.write("# " + num_vertices + " vertices\n")
    for v in vertices:
        tet_output.write("v " + v + "\n")
    tet_output.write("\n")
    tet_output.write("# " + num_tetrahedra + " tetrahedra\n")
    for t in tetrahedra:
        line = t.split(' 0')[0]
        line = line.split(" ")
        line = [str(int(k) - 1) for k in line]
        l_text = ' '.join(line)
        tet_output.write("t " + l_text + "\n")  
        
        
def get_rays(cylinder_shift, cylinder_radius, cylinder_length, num_rays=1024):

    # Create target cylinder
    quat = [0.5, 0.5,0.5, 0.5]
    trans_mat = transformations.quaternion_matrix(quat)
    trans_mat[:3,3] = np.array([0, -0.5, 0.04]) + cylinder_shift
    # cylinder_mesh = trimesh.creation.annulus(r_min=0.0149*0.6, r_max=0.015*0.6, height=0.1, transform = trans_mat)
    # cylinder_mesh = trimesh.creation.annulus(r_min=0.01498*0.6, r_max=0.015*0.6, height=0.1*0.7, transform = trans_mat)
    cylinder_mesh = trimesh.creation.annulus(r_min=cylinder_radius-0.00002, r_max=cylinder_radius, 
                                             height=cylinder_length*0.7, transform = trans_mat)


    # Sample points (and corresponding normals) on mesh
    pc, faces = trimesh.sample.sample_surface_even(cylinder_mesh, count=num_rays)   # might not sample enough num_rays points
    normals = cylinder_mesh.face_normals[faces]

    ray_origins, ray_directions = pc, normals/12    # /25: shorten normals for visualization

    # stack rays into line segments for visualization as Path3D
    ray_visualize = trimesh.load_path(np.hstack((
        ray_origins,
        ray_origins + ray_directions)).reshape(-1, 2, 3))

    return ray_origins, ray_directions, ray_visualize, cylinder_mesh



def compute_intersection_percent(final_full_pc, tri_indices, cylinder_shift, 
                                 cylinder_radius, cylinder_length,
                                 vis = False, num_rays=1024):
    
    ray_origins, ray_directions, ray_visualize, cylinder_mesh = get_rays(cylinder_shift, cylinder_radius, cylinder_length, num_rays)

    tissue_mesh = trimesh.Trimesh(vertices=final_full_pc,
                            faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))

    ### run the mesh- ray test
    locations, index_ray, index_tri = tissue_mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions)
    intersection = trimesh.points.PointCloud(locations)

    index_ray = set(index_ray)
    # print("number of intersections:", len(index_ray))
    # print("Percent coverage:", len(index_ray)/ray_origins.shape[0])    


    if vis:
        tissue_mesh.visual.face_colors = [255, 0, 0,200]
        scene = trimesh.Scene([
            cylinder_mesh, tissue_mesh,
            intersection,
            ray_visualize])
        scene.show()


    return len(index_ray)/ray_origins.shape[0]

def read_pickle_data(data_path):
    with open(data_path, 'rb') as handle:
        return pickle.load(handle)      

def write_pickle_data(data, data_path):
    with open(data_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)    

def pcd_ize(pc, color=None, vis=False):
    """ 
    Convert point cloud numpy array to an open3d object (usually for visualization purpose).
    """
    import open3d
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)   
    if color is not None:
        pcd.paint_uniform_color(color) 
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd