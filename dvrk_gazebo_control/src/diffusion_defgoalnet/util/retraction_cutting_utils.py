import numpy as np

def get_y_to_x_ratio(start_point, end_point):
    theta = np.arctan((end_point[1] - start_point[1]) / (end_point[0] - start_point[0]))
    print("theta:", theta)
    return [-np.tan(np.pi/4 - theta), np.tan(np.pi/4 + theta)]