import pickle

import sys, os
sys.path.extend([
    os.path.join('pybullet-planning')
])
from pybullet_tools.utils import load_model, connect, create_plane, TAN, get_joints, get_joint_names, \
    multiply, clone_body, get_link_subtree, get_relative_pose
import numpy as np
import mayavi.mlab as mlab
from os.path import join
from pointcloud_processing import get_bucket_pc

visualize_filename = "0.pkl"

start_arm_joint = 18
start_base_joint = 12

num_arm_joints = 10
num_base_joints = 3

base_joint_indices = list(range(start_base_joint, start_base_joint+num_base_joints))
arm_joint_indices = list(range(start_arm_joint, start_arm_joint+num_arm_joints))

return_order = ['root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint', 'linear_actuator_height', 'right_panda_joint1', 'right_panda_joint2', 'right_panda_joint3', 'right_panda_joint4', 'right_panda_joint5', 'right_panda_joint6', 'right_panda_joint7', 'right_panda_finger_joint1', 'right_panda_finger_joint2']
return_order_lookup_dict = {name: index for name, index in zip(return_order, base_joint_indices+arm_joint_indices)}


connect(use_gui=False)
model_path = join(os.path.abspath(os.getcwd()), 'models/robot/sciurus/A2_single.urdf')
robot = load_model(model_path)
floor = create_plane(color=TAN)
robot_joints = get_joints(robot)
joint_names = get_joint_names(robot, robot_joints)
joint_lookup_dict = {name: joint for name, joint in zip(joint_names, robot_joints)}

with open('temp/{}'.format(visualize_filename), 'rb') as handle:
    obs = pickle.load(handle)


# Visualize
class GPDGrasp():
    def __init__(self, float_list):
        self.pos = float_list[:3]
        self.quat = float_list[3:7]
        self.score = float_list[7]
        self.binormal = float_list[8:11]
        self.approach = float_list[11:14]

    def getPosition(self):
        return np.array(self.pos)

    def getBinormal(self):
        return np.array(self.binormal)

    def getApproach(self):
        return np.array(self.approach)

    def getQuat(self):
        return np.array(self.quat)


def clone_gripper(robot):
    panda_hand_link = name_link_dict["right_panda_hand"]
    subtree = get_link_subtree(robot, panda_hand_link)
    component = clone_body(robot, links=subtree, visual=False, collision=True)
    return component

def get_grasp(grasp_tool, gripper_from_tool):
    return multiply(gripper_from_tool, grasp_tool)

def get_parent_from_tool(robot):
    tool_link = name_link_dict["panda_tool_tip"]
    parent_link = name_link_dict["right_panda_link8"]
    return get_relative_pose(robot, tool_link, parent_link)



pc, bucket_pc, pc_color, bucket_pc_color = get_bucket_pc(obs)
rgba = np.zeros((bucket_pc.shape[0], 4), dtype=np.uint8)
rgba[:, :3] = np.asarray(bucket_pc_color)*255.0
rgba[:, 3] = 255
src = mlab.pipeline.scalar_scatter(bucket_pc[:, 0], bucket_pc[:, 1], bucket_pc[:, 2])
src.add_attribute(rgba, 'colors')
src.data.point_data.set_active_scalars('colors')
g = mlab.pipeline.glyph(src)
g.glyph.scale_mode = "data_scaling_off"
g.glyph.glyph.scale_factor = 0.01
mlab.show()