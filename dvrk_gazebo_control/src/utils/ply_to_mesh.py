import os

os.chdir('/home/baothach/fTetWild/build') 



object_meshes_path = "/home/baothach/sim_data/BigBird/BigBird_mesh"

# object_name = "cholula_chipotle_hot_sauce"

# for object_name in sorted(os.listdir(object_meshes_path)):
for object_name in os.listdir(object_meshes_path):
    mesh_path = object_meshes_path + "/" + object_name + "/meshes/poisson.ply"
    save_fTetwild_mesh_path = object_meshes_path + "/" + object_name + "/meshes/poisson.mesh"
    os.system("./FloatTetwild_bin -o " + save_fTetwild_mesh_path + " -i " + mesh_path)