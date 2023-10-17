import os
import pickle
import numpy as np

urdf_main_path = "/home/baothach/sim_data/Custom/Custom_objects/random_stuff/urdf"
object_name = "chicken_breast"

save_urdf_path = os.path.join(urdf_main_path, object_name)
os.makedirs(save_urdf_path, exist_ok=True)


density = 100
poissons = 0.3
scale = 1
attach_dist = 0.01


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

    </robot>
    """
        f.write(urdf_str)
        f.close()

