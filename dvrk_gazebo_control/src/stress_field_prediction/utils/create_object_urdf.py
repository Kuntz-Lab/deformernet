import os
import pickle

# object_mesh_path = "/home/baothach/sim_data/Custom/Custom_mesh/retraction_cutting/multi_box"
# base_mesh_path = "/home/baothach/sim_data/Custom/Custom_mesh/retraction_cutting/multi_box"
# object_urdf_path = "/home/baothach/sim_data/Custom/Custom_urdf/retraction_cutting/multi_box"

object_name = "square"
object_mesh_path = f'/home/baothach/sim_data/stress_prediction_data/objects/{object_name}'
object_urdf_path = object_mesh_path

os.makedirs(object_urdf_path,exist_ok=True)

density = 1000
youngs = 1e4
poissons = 0.3
attach_dist = 0.0
scale = 1


   
cur_urdf_path = object_urdf_path + "/soft_body.urdf"
f = open(cur_urdf_path, 'w')

urdf_str = f"""<?xml version="1.0" encoding="utf-8"?>    

<robot name="{object_name}">
    <link name="{object_name}">    
        <fem>
            <origin rpy="0.0 0.0 0.0" xyz="0 0 0" />
            <density value="{density}" />
            <youngs value="{round(youngs)}"/>
            <poissons value="{poissons}"/>
            <damping value="0.0" />
            <attachDistance value="{attach_dist}"/>
            <tetmesh filename="{os.path.join(object_mesh_path, object_name+".tet")}"/>
            <scale value="{scale}"/>
        </fem>
    </link>

</robot>
"""

f.write(urdf_str)
f.close()

