#!/usr/bin/env python3


"""Perform a grasp evaluation on a soft object using a Panda hand.

Usage example:
python3 main.py --object=rectangle --grasp_ind=3 --youngs=2e5 --density=1000
    --ori_start=10 --ori_end=10 --mode=reorient
"""

import argparse

from grasp_evaluator import GraspEvaluator
import rospy

# Create command line flag options
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--object', required=True, help="Name of object")
parser.add_argument('--grasp_ind', default=0, type=int, help="Index of grasp candidate to test")
parser.add_argument(
    '--ori_start',
    default=0,
    type=int,
    help="Start index of vectors to test. [0, 15]")
parser.add_argument(
    '--ori_end',
    default=5,
    type=int,
    help="End index of vectors to test. [0, 15]")
parser.add_argument('--density', default='1000', type=str, help="Density of object [kg/m^3]")
parser.add_argument('--youngs', default='5e5', type=str, help="Elastic modulus of object [Pa]")
parser.add_argument('--poissons', default='0.3', type=str, help="Poisson's ratio of object")
parser.add_argument('--friction', default='0.7', type=str, help="Dynamic friction")
parser.add_argument(
    '--mode',
    default='pickup',
    type=str,
    help="Name of test to run, one of {pickup, reorient, shake, twist, squeeze_no_gravity}")
parser.add_argument(
    '--tag',
    default='',
    type=str,
    help="Additional string to add onto name of results files.")
args = parser.parse_args()

oris = [args.ori_start, args.ori_end]



if __name__ == "__main__":
    
    rospy.init_node('stress_prediction')
    
    grasp_evaluator = GraspEvaluator(args.object, args.grasp_ind, oris, args.density,
                                     args.youngs, args.poissons, args.friction, args.mode, args.tag)

    if not grasp_evaluator.data_exists:
        grasp_evaluator.run_simulation()
