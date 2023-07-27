import numpy as np
import transformations


quat = [0.0, 0.0, 0.707107, 0.707107]
eulers = transformations.euler_from_quaternion(quat)
print("eulers:", eulers)