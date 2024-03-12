import os
import pickle

save_urdf_path = "/home/baothach/sim_data/Custom/Custom_urdf/multi_cylinder_1kPa"
object_meshes_path = "/home/baothach/sim_data/Custom/Custom_mesh/multi_cylinder_1kPa"
shape_name = "cylinder"

density = 100
# youngs = 1e3
poissons = 0.3
scale = 0.5
attach_dist = 0.01



with open(os.path.join(object_meshes_path, "primitive_dict_cylinder.pickle"), 'rb') as handle:
    data = pickle.load(handle)

for i in range(0,100):
    object_name = shape_name + "_" + str(i)
    radius = data[object_name]["radius"]
    height = data[object_name]["height"]
    youngs = round(data[object_name]["youngs"])

    # radius = 0.08
    # height = 0.2
    # youngs = 3000

    cur_urdf_path = save_urdf_path + '/' + object_name + '.urdf'
    f = open(cur_urdf_path, 'w')
    if True:
        urdf_str = """    
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
                <origin xyz="0.0 0.0 """ +str(height/4.0 + 10.0) + """\"/>
                <geometry>
                    <box size="0.15 """ + str(radius) + """ 0.005\"/>
                </geometry>
            </visual>
            <collision>
                <origin xyz="0.0 0.0 """ +str(height/4.0) + """\"/>
                <geometry>
                    <box size="0.15 """ + str(radius) + """ 0.005\"/>
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

