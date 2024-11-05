import os
import pickle
import numpy as np


def get_extents_object(tet_file):
    """Return [min_x, min_y, min_z], [max_x, max_y, max_z] for a tet mesh"""
    mesh_lines = list(open(tet_file, "r"))
    mesh_lines = [line.strip('\n') for line in mesh_lines]
    zs = []
    particles = []
    for ml in mesh_lines:
        sp = ml.split(" ")
        if sp[0] == 'v':
            particles.append([float(sp[j]) for j in range(1,4)])
                
    particles = np.array(particles)
    xs = particles[:,0]
    ys = particles[:,1]
    zs = particles[:,2]
    
    return [[min(xs), min(ys), min(zs)],\
            [max(xs), max(ys), max(zs)]]   
    

urdf_main_path = "/home/baothach/sim_data/Custom/Custom_objects/random_stuff/urdf"
object_name = "chicken_breast"


save_urdf_path = os.path.join(urdf_main_path, f"{object_name}_attached")
os.makedirs(save_urdf_path, exist_ok=True)


mesh_path = "/home/baothach/sim_data/Custom/Custom_objects/random_stuff/mesh"
tet_file = os.path.join(mesh_path, f"{object_name}.tet")
extents = get_extents_object(tet_file)
height = round(np.array(extents[1][1])-np.array(extents[0][1]), 2)
thickness = round(np.array(extents[1][2])-np.array(extents[0][2]), 2)
width = round(np.array(extents[1][0])-np.array(extents[0][0]), 2)

print("height, width, thickness", height, width, thickness)

density = 100
poissons = 0.3
scale = 1
attach_dist = 0.01  #0.01


for i in range(100):

    youngs = round(np.random.uniform(low=1000, high=10000))

    cur_urdf_path = f"{save_urdf_path}/{object_name}_{i}.urdf"
    
    f = open(cur_urdf_path, 'w')
    if True:
        urdf_str = """<?xml version="1.0" encoding="utf-8"?>        
    <robot name=\"""" + object_name + """\">
        <link name=\"""" + object_name + """\">    
            <fem>
                <origin rpy="0.0 0.0 0.0" xyz="0 0 0" />
                <density value=\"""" + str(density) + """\" />
                <youngs value=\"""" + str(youngs) + """\"/>
                <poissons value=\"""" + str(poissons) + """\" />
                <damping value="0.0" />
                <attachDistance value=\"""" + str(attach_dist) + """\" />
                <tetmesh filename=\"../../mesh/""" + str(object_name) + """.tet\" />
                <scale value=\"""" + str(scale) + """\"/>
            </fem>
        </link>

        <link name="fix_frame">
            <visual>
                <origin xyz=\"0.0 """ +str(-height * 0.4 - 10.0) + """ 0.0\"/>              
                <geometry>
                    <box size="0.15 0.005 """ + str(thickness) + """\"/>
                </geometry>
            </visual>
            <collision>
                <origin xyz=\"0.0 """ +str(-height * 0.4) + """ 0.0\"/>              
                <geometry>
                    <box size="0.15 0.005 """ + str(thickness) + """\"/>
                </geometry>
            </collision>
            <inertial>
                <mass value="500000"/>
                <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
            </inertial>
        </link>
        
        <joint name = "attach" type = "fixed">
            <origin xyz = "0.0 0.0 0.0" rpy = "0 0 0"/>
            <parent link =\"""" + object_name + """\"/>
            <child link = "fix_frame"/>
        </joint>

    </robot>
    """
        f.write(urdf_str)
        f.close()

