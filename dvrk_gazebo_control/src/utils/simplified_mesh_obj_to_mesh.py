import os

os.chdir('/home/baothach/fTetWild/build') 



object_meshes_path = "/home/baothach/sim_data/Custom/Custom_mesh"

# object_name = "cholula_chipotle_hot_sauce"

# for object_name in sorted(os.listdir(object_meshes_path)):
count = 0
for object_name in sorted(os.listdir(object_meshes_path)):
    print("--------------------------------------")
    print("***DOING OBJECT: ", object_name)
    mesh_path = object_meshes_path + "/" + object_name + "/simplified_mesh.obj"
    save_fTetwild_mesh_path = object_meshes_path + "/" + object_name + "/simplified_mesh.mesh"
    os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path)

    # count += 1
    # if count >= 1:
    #     break