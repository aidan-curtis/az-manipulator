import pickle
import pybullet as p

import sys, os
sys.path.extend([
    os.path.join('pybullet-planning'),
    os.path.join('..', '..', 'pybullet-planning')
])
from pybullet_tools.utils import load_pybullet, load_model, connect, create_plane, TAN, get_joints, get_joint_names, \
    set_joint_positions, get_links, get_link_names, get_link_pose, multiply, clone_body, get_link_subtree, \
    set_pose, Pose, Point, get_relative_pose, invert, remove_body, HideOutput, HideOutput, disconnect, get_movable_joints, \
    get_joint_positions
import math
import numpy as np
import copy
from os.path import join
from scipy.spatial.transform import Rotation as R


rainbow = [(192, 57, 43), (211, 84, 0), (243, 156, 18), (22, 160, 133),
           (39, 174, 96), (41, 128, 185), (142, 68, 173)]

def get_color(index, total):
    def middle(lower, upper, percentage):
        return lower + percentage * (upper - lower)
    interval = math.ceil(total / (len(rainbow)-1))
    i = index // interval
    shift = index % interval / interval
    return tuple([middle(rainbow[i][j], rainbow[i+1][j], shift)/255 for j in range(3)])

def get_time():
    import datetime
    now = datetime.datetime.now()
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    second = '{:02d}'.format(now.second)
    microsecond = '{:02d}'.format(now.microsecond)
    return '{}-{}-{}:{}:{}.{}'.format(month, day, hour, minute, second, microsecond)


def dist(p1, p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))
def dist3d(p1, p2):
    assert len(p1)==3
    return math.sqrt(sum([(p1[i]-p2[i])**2 for i in range(3)]))



# Visualize
class GPDGrasp():
    def __init__(self, float_list):
        self.pos = float_list[:3]
        self.quat = float_list[3:7]
        self.score = float_list[7]
        rotation_matrix = R.from_quat(self.quat).as_matrix()
        self.binormal = rotation_matrix[:, 1]
        self.approach = rotation_matrix[:, 0]
        self.axis = rotation_matrix[:, 2]

    def getPosition(self):
        return np.array(self.pos)

    def getAxis(self):
        return np.array(self.axis)

    def getBinormal(self):
        return np.array(self.binormal)

    def getApproach(self):
        return np.array(self.approach)

    def getQuat(self):
        return np.array(self.quat)

def clone_gripper(robot):
    links = get_links(robot)
    name_link_dict = {name: link for name, link in zip(get_link_names(robot, links), links)}
    panda_hand_link = name_link_dict["right_panda_hand"]  ##
    subtree = get_link_subtree(robot, panda_hand_link)
    component = clone_body(robot, links=subtree, visual=False, collision=True)
    return component

def get_grasp(grasp_tool, gripper_from_tool):
    return multiply(gripper_from_tool, grasp_tool)

def get_parent_from_tool(robot):
    links = get_links(robot)
    name_link_dict = {name: link for name, link in zip(get_link_names(robot, links), links)}
    tool_link = name_link_dict["right_panda_tool_tip"]
    parent_link = name_link_dict["right_panda_link8"]
    return get_relative_pose(robot, tool_link, parent_link)

def visualize_saved_pointcloud(visualize_filename = "0.pkl", num=1000, VIZ=False, window_size=(600,600), env="OpenCabinetDrawer"):
    with open('temp/{}'.format(visualize_filename), 'rb') as handle:
        obs = pickle.load(handle)
    get_grasps(obs, num=num, VIZ=VIZ, window_name=visualize_filename.replace('.pkl',''), window_size=window_size, env=env)

if __name__ == "__main__":

    # print(get_saved_chair_traj("0.pkl"))
    # visualize_saved_pointcloud("0.pkl", num=1000, VIZ=True)

    visualize_saved_pointcloud("0.pkl", num=1000, VIZ=True, env="MoveBucket")

    ## visualize point clouds for all evaluation environments
    # pointclouds = [f for f in listdir('temp') if '.pkl' in f and '0.pkl' not in f]
    # pointclouds.sort()
    # for pc in pointclouds:
    #     visualize_saved_pointcloud(pc, num=1000, VIZ=True)  ## , window_size=(2560, 1440)
    #     break