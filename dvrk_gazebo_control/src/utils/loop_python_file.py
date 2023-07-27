import os
import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')

i = 0
while i < 10:
    os.pause(2)
    os.system("test_stuff/nothing.py")
    i += 1