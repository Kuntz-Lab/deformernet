
import sys
sys.path.append('../')
from util.retraction_cutting_utils import get_eef_position



# Example usage:
s_point = [1, -2]
e_point = [6, -6]
alpha_angle = 60  # Angle in degrees
magnitude_length = 3  # Desired magnitude

# Adjust the placement and visualize the result by setting vis=True
a_point, b_point, midpoint = get_eef_position(s_point, e_point, alpha_angle, magnitude_length, vis=True)
