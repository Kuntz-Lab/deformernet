import os
import pickle5 as pickle

# save_urdf_path = "/home/baothach/sim_data/Custom/Custom_urdf/multi_hemis_1000Pa"
# object_meshes_path = "/home/baothach/sim_data/Custom/Custom_mesh/multi_hemis_1000Pa"
# save_urdf_path = "/home/baothach/sim_data/Custom/Custom_urdf/multi_hemis_1000Pa_2"
# object_meshes_path = "/home/baothach/sim_data/Custom/Custom_mesh/multi_hemis_1000Pa_2"
# save_urdf_path = "/home/baothach/shape_servo_data/evaluation/urdf/hemis_1k/inside_sample_ratio"
# object_meshes_path = "/home/baothach/shape_servo_data/evaluation/meshes/hemis_1k/inside_sample_ratio"
# save_urdf_path = "/home/baothach/sim_data/Custom/Custom_urdf/multi_hemis_10kPa"
# object_meshes_path = "/home/baothach/sim_data/Custom/Custom_mesh/multi_hemis_10kPa"
save_urdf_path = "/home/baothach/shape_servo_data/evaluation/urdf/hemis_1k/inside"
object_meshes_path = "/home/baothach/shape_servo_data/evaluation/meshes/hemis_1k/inside"
os.makedirs(save_urdf_path,exist_ok=True)


shape_name = "hemis"

density = 100
# youngs = 1e3
poissons = 0.3
scale = 0.5
attach_dist = 0.04 #0.05



# with open(os.path.join(object_meshes_path, "primitive_dict_hemis.pickle"), 'rb') as handle:
#     data = pickle.load(handle)

for i in range(0,10):
    # object_name = shape_name + "_" + str(i)
    # radius = data[object_name]["radius"]
    # origin = data[object_name]["origin"]
    # youngs = round(data[object_name]["youngs"])

    object_name = shape_name + "_" + str(i)
    with open(os.path.join(object_meshes_path, object_name + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)
    radius = data["radius"]
    origin = data["origin"]
    youngs = round(data["youngs"])

    # height = 0.4
    # width = 0.15
    # thickness = 0.08
    # youngs = 3000

    cur_urdf_path = save_urdf_path + '/' + object_name + '.urdf'
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
                <tetmesh filename=\"""" + object_meshes_path + """/""" + str(object_name) + """.tet\" />
                <scale value=\"""" + str(scale) + """\"/>
            </fem>
        </link>

        <link name="fix_frame">
            <visual>
                <origin xyz="0 """ +str(-radius/2. - 10.0) + """ """ + str((radius-origin)/2) + """\"/>              
                <geometry>
                    <box size="0.3 0.005 """ + str((radius-origin)/2) + """\"/>
                </geometry>
            </visual>
            <collision>
                <origin xyz="0 """ +str(-radius/2.) + """ """ + str((radius-origin)/2) + """\"/>              
                <geometry>
                    <box size="0.3 0.005 """ + str((radius-origin)/2) + """\"/>
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

