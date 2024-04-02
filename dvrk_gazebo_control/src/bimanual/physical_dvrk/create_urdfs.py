import os
import pickle


urdf_path = "/home/baothach/sim_data/Custom/Custom_urdf/physical_dvrk/multi_box_10kPa"
os.makedirs(urdf_path,exist_ok=True)

mesh_path = "/home/baothach/sim_data/Custom/Custom_mesh/physical_dvrk/multi_box_10kPa"
mesh_relative_path = "/".join(mesh_path.split("/")[-3:])

shape_name = "box"

density = 100
# youngs = 1e3
poissons = 0.3
scale = 1.0
attach_dist = 0.00


with open(os.path.join(mesh_path, "primitive_dict_box.pickle"), 'rb') as handle:
    data = pickle.load(handle)

for i in range(100):

    object_name = shape_name + "_" + str(i)
    height = data[object_name]["height"]
    width = data[object_name]["width"]
    thickness = data[object_name]["thickness"]
    youngs = round(data[object_name]["youngs"])

    cur_urdf_path = urdf_path + '/' + shape_name + "_" + str(i) + '.urdf'
    
    f = open(cur_urdf_path, 'w')
    urdf_str = f"""<?xml version="1.0" encoding="utf-8"?>    
    
    <robot name="{object_name}">
        <link name="{object_name}">    
            <fem>
                <origin rpy="0.0 0.0 0.0" xyz="0 0 0" />
                <density value="{density}" />
                <youngs value="{youngs}"/>
                <poissons value="{poissons}"/>
                <damping value="0.0" /> 
                <attachDistance value="{attach_dist}"/>
                <tetmesh filename="../../../{mesh_relative_path}/{object_name+".tet"}"/>
                <scale value="{scale}"/>
            </fem>
        </link>
    </robot>
    """

    f.write(urdf_str)
    f.close()

