import pickle
import numpy as np
import os
import pickle



data_recording_path = "/home/baothach/shape_servo_data/generalization/plane/results" 

num_success = 0
num_fail = 0
for i in range(0,100):
    with open(os.path.join(data_recording_path, f"plane_{i}.pickle"), 'rb') as handle:
        data = pickle.load(handle) 
    if data["success"]:
        num_success += 1
    else:
        num_fail += 1

print("success:", num_success)
print("Fail:", num_fail)