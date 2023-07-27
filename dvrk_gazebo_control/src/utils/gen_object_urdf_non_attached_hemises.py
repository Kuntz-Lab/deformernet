import os
import pickle5 as pickle

# save_urdf_path = '/home/baothach/shape_servo_data/evaluation/bimanual_urdf/hemis_1k/inside'
# object_meshes_path = '/home/baothach/shape_servo_data/evaluation/meshes/hemis_1k/inside'
# save_urdf_path = "/home/baothach/sim_data/Custom/Custom_urdf/bimanual_multi_hemis_5kPa"
# object_meshes_path = "/home/baothach/sim_data/Custom/Custom_mesh/multi_hemis_5kPa" 
save_urdf_path = "/home/baothach/shape_servo_data/evaluation/bimanual_urdf/hemis_5k/inside"
object_meshes_path = "/home/baothach/shape_servo_data/evaluation/meshes/hemis_5k/inside"

os.makedirs(save_urdf_path,exist_ok=True)

shape_name = "hemis"

density = 100
# youngs = 1e3
poissons = 0.3
scale = 0.5
attach_dist = 0.04 #0.05


# with open(os.path.join(object_meshes_path, "primitive_dict_hemis.pickle"), 'rb') as handle:
#     data = pickle.load(handle)

for i in range(100):
    # object_name = shape_name + "_" + str(i)
    
    # radius = data[object_name]["radius"]
    # origin = data[object_name]["origin"]
    # youngs = round(data[object_name]["youngs"])

    # # object_name = shape_name + "_" + str(i)
    # # with open(os.path.join(object_meshes_path, object_name + ".pickle"), 'rb') as handle:
    # #     data = pickle.load(handle)
    # # radius = data["radius"]
    # # origin = data["origin"]
    # # youngs = round(data["youngs"])

    # # height = 0.2
    # # width = 0.2
    # # thickness = 0.04
    # # youngs = 4000

    object_name = shape_name + "_" + str(i%10)
    with open(os.path.join(object_meshes_path, object_name + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)
    radius = data["radius"]
    origin = data["origin"]
    youngs = round(data["youngs"])


    # cur_urdf_path = save_urdf_path + '/' + object_name + '.urdf'
    cur_urdf_path = save_urdf_path + '/' + shape_name + "_" + str(i) + '.urdf'
    
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

    </robot>
    """
        f.write(urdf_str)
        f.close()
