from custom_controllers import LPFilter, PIDController, PositionController

import numpy as np
import gym
import mani_skill.env
from gym.spaces import Box
import gym
from os.path import join
import os
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
from pointcloud_utils import get_parent_from_tool, clone_gripper, GPDGrasp
import pybullet as p
import time
from controller import get_robot_qpos_from_obs
from pointcloud_utils import dist, dist3d

class BasePolicy(object):
    def __init__(self, opts=None):
        self.obs_mode = 'pointcloud'

    def act(self, observation):
        raise NotImplementedError()

    def reset(self): # if you use an RNN-based policy, you need to implement this function
        pass

class RandomPolicy(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        env = gym.make(env_name)
        self.action_space = copy.copy(env.action_space)
        env.close()
        del env

    def act(self, state):
        return self.action_space.sample()

class APolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        super(APolicy).__init__(*args, **kwargs)

    def init_position_controller(self, velocity_limit, kp, ki, kd):
        control_frequency = 100
        cutoff_frequency = 40

        lpfilter = LPFilter(control_frequency, cutoff_frequency)
        pid_controller = PIDController(kp, ki, kd, control_frequency, velocity_limit)
        position_controller = PositionController(pid_controller, lpfilter)
        return position_controller

    def get_pc(self, obs):
        raise NotImplementedError

    def generate_grasps(self, segmented_pc):
        raise NotImplementedError

    def get_conf_data(self, grasp_pose, robots, name_link_dict, q0):
        raise NotImplementedError

    def get_grasps(self, obs, num=5, VIZ=False, window_name='grasps', window_size=(600, 600), only_pose=False, downsample=1):

        q0 = get_robot_qpos_from_obs(obs)
        joint_position_vals = obs['agent']
        robots, return_order, return_order_lookup_dict = self.get_pb_robot()

        # Center
        robot_joints = get_joints(robots['center'])
        joint_names = get_joint_names(robots['center'], robot_joints)
        joint_lookup_dict = {name: joint for name, joint in zip(joint_names, robot_joints)}

        pb_joint_indices = [joint_lookup_dict[joint_name] for joint_name in return_order]
        pb_joint_values = [joint_position_vals[return_order_lookup_dict[joint_name]] for joint_name in return_order]

        set_joint_positions(robots['center'], pb_joint_indices, pb_joint_values)

        links = get_links(robots['center'])
        name_link_dict = {name: link for name, link in zip(get_link_names(robots['center'], links), links)}

        pc, grasp_pc, pc_color, _ = self.get_pc(obs)
        grasps = self.generate_grasps(grasp_pc)

        if VIZ:
            import mayavi.mlab as mlab
            fig = mlab.figure(window_name, size=window_size)
            rgba = np.zeros((pc.shape[0], 4), dtype=np.uint8)
            rgba[:, :3] = np.asarray(pc_color) * 255.0
            rgba[:, 3] = 255
            src = mlab.pipeline.scalar_scatter(pc[:, 0], pc[:, 1], pc[:, 2])
            src.add_attribute(rgba, 'colors')
            src.data.point_data.set_active_scalars('colors')
            g = mlab.pipeline.glyph(src)
            g.glyph.scale_mode = "data_scaling_off"
            g.glyph.glyph.scale_factor = 0.01

        parent_from_tool = get_parent_from_tool(robots['center'])
        valid_grasps = []
        total = min(len(grasps), num)
        count = -1

        if(len(grasps)>50):
            using_grasps = grasps[::downsample]
        else:
            using_grasps = grasps

        for grasp_i, grasp_candidate in enumerate(using_grasps):
            count += 1
            # Create a copy of the pybullet gripper
            new_gripper = clone_gripper(robots['center'])

            # Move it to the correct gripper pose
            grasp_candidate_pose = (grasp_candidate[:3], grasp_candidate[3:7])
            grasp_pose = multiply(grasp_candidate_pose, invert(parent_from_tool))


            set_pose(new_gripper, grasp_pose)

            # TODO: Set gripper to open position
            # set_joint_positions(gripper, robot.get_component_joints(gripper_group), open_conf)

            # Create rays for each point
            ray_from_positions = []
            ray_to_positions = []
            dist = 0.005
            relevant_dist = 0.2
            rpc = []
            for i in range(pc.shape[0]):
                if (np.linalg.norm(pc[i] - grasp_pose[0]) < relevant_dist):
                    rpc.append(pc[i])

            rpc = np.array(rpc)
            ray_test_results = \
                p.rayTestBatch(rpc, [[rpc[i][0], rpc[i][1], rpc[i][2] - dist] for i in range(rpc.shape[0])]) + \
                p.rayTestBatch(rpc, [[rpc[i][0], rpc[i][1], rpc[i][2] + dist] for i in range(rpc.shape[0])]) + \
                p.rayTestBatch(rpc, [[rpc[i][0], rpc[i][1] - dist, rpc[i][2]] for i in range(rpc.shape[0])]) + \
                p.rayTestBatch(rpc, [[rpc[i][0], rpc[i][1] + dist, rpc[i][2]] for i in range(rpc.shape[0])]) + \
                p.rayTestBatch(rpc, [[rpc[i][0] - dist, rpc[i][1], rpc[i][2]] for i in range(rpc.shape[0])]) + \
                p.rayTestBatch(rpc, [[rpc[i][0] + dist, rpc[i][1], rpc[i][2]] for i in range(rpc.shape[0])])

            # remove_body(new_gripper)
            if new_gripper not in [ray_test_result[0] for ray_test_result in ray_test_results]:
                # Load in the gripper
                hand = GPDGrasp(grasp_candidate)
                outer_diameter = 0.10
                hw = 0.5 * outer_diameter
                base_depth = 0.02
                finger_width = 0.01
                hand_depth = 0.06

       
                left_bottom = hand.getPosition() - (hw - 0.5 * finger_width) * hand.getBinormal()
                right_bottom = hand.getPosition() + (hw - 0.5 * finger_width) * hand.getBinormal()
                left_center = left_bottom + 0.5 * hand_depth * hand.getApproach()
                right_center = right_bottom + 0.5 * hand_depth * hand.getApproach()
                base_center = left_bottom + 0.5 * (right_bottom - left_bottom) - 0.01 * hand.getApproach()
                approach_center = base_center - 0.04 * hand.getApproach()
                center_bottom = (np.array(left_bottom) + np.array(right_bottom)) / 2.0
                first = (np.array(left_bottom) + np.array(right_bottom)) / 2.0 - hand_depth * hand.getApproach()
                quat = hand.getQuat()

                grasp_pc = np.array([left_center, left_bottom, right_bottom, right_center])
                handle = np.array([center_bottom, first])

                if VIZ:
                    gripper_color = get_color(count, total) ## (0.1, 0.1, 0.9)
                    mlab.plot3d(grasp_pc[:, 0], grasp_pc[:, 1], grasp_pc[:, 2], color=gripper_color, tube_radius=0.003,
                                opacity=1)
                    mlab.plot3d(handle[:, 0], handle[:, 1], handle[:, 2], color=gripper_color, tube_radius=0.003, opacity=1)            

                if(not only_pose):
                    valid_grasps.append(self.get_conf_data(grasp_pose, robots, name_link_dict, q0))
                else:

                    robot = robots['center']
                    # Get the base pose of the left and right arm
                    body_link_pose = get_link_pose(robot, name_link_dict["body_link"])
                    left_link_pose = get_link_pose(robot, name_link_dict["left_panda_hand"])
                    right_link_pose = get_link_pose(robot, name_link_dict["right_panda_hand"])   
                    left_dist = dist3d(left_link_pose[0], grasp_pose[0])
                    right_dist = dist3d(right_link_pose[0], grasp_pose[0])
                    valid_grasps.append( (grasp_pose, left_dist, right_dist) )

                # time.sleep(10)
            # if len(valid_grasps) >= num:
            #     break

        if VIZ:
            mlab.show()
        return valid_grasps

class DualAPolicy(APolicy):

    def __init__(self, *args, **kwargs):
        super(DualAPolicy).__init__(*args, **kwargs)

    def get_pb_robot(self):
        start_base_joint = 24
        start_arm_joint = start_base_joint+6


        num_arm_joints = 19
        num_base_joints = 3

        base_joint_indices = list(range(start_base_joint, start_base_joint + num_base_joints))
        arm_joint_indices = list(range(start_arm_joint, start_arm_joint + num_arm_joints))

        #if sys.stdin and sys.stdin.isatty(): ## running in command line
        model_path = join(os.path.abspath(os.getcwd()), 'models/robot/sciurus/A2.urdf')
        #else:  ## running in IDE
        #    model_path = join(os.path.abspath(os.getcwd()), '../../models/robot/sciurus/A2.urdf')
        return_order = ['root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint', 'linear_actuator_height', 
                        'right_panda_joint1', 'right_panda_joint2', 'right_panda_joint3', 'right_panda_joint4', 
                        'right_panda_joint5', 'right_panda_joint6', 'right_panda_joint7', 'right_panda_finger_joint1', 'right_panda_finger_joint2', 
                        'left_panda_joint1', 'left_panda_joint2', 'left_panda_joint3', 'left_panda_joint4', 
                        'left_panda_joint5', 'left_panda_joint6', 'left_panda_joint7', 'left_panda_finger_joint1', 'left_panda_finger_joint2']

        return_order_lookup_dict = {name: index for name, index in
                                    zip(return_order, base_joint_indices + arm_joint_indices)}

        connect(use_gui=False)
        with HideOutput(enable=True):
            # print(model_path)
            robot = load_pybullet(model_path)
            floor = create_plane(color=TAN)

        robots = {
            "center": robot,
        }
        return robots, return_order, return_order_lookup_dict


class SingleAPolicy(APolicy):
    
    def __init__(self, *args, **kwargs):
        super(DualAPolicy).__init__(*args, **kwargs)

    def get_pb_robot(self):
        start_base_joint = 12
        start_arm_joint = start_base_joint+6

        num_arm_joints = 10
        num_base_joints = 3

        base_joint_indices = list(range(start_base_joint, start_base_joint + num_base_joints))
        arm_joint_indices = list(range(start_arm_joint, start_arm_joint + num_arm_joints))

        model_path = join(os.path.abspath(os.getcwd()), 'models/robot/sciurus/A2_single.urdf')
        model_path = join(os.path.dirname(os.path.abspath(__file__)), '../models/robot/sciurus/A2_single.urdf')
        return_order = ['root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint', 'linear_actuator_height',
                    'right_panda_joint1', 'right_panda_joint2', 'right_panda_joint3', 'right_panda_joint4',
                    'right_panda_joint5', 'right_panda_joint6', 'right_panda_joint7', 'right_panda_finger_joint1',
                    'right_panda_finger_joint2']
        return_order_lookup_dict = {name: index for name, index in
                                    zip(return_order, base_joint_indices + arm_joint_indices)}

        connect(use_gui=False)
        with HideOutput(enable=True):
            # print(model_path)
            robot = load_model(model_path)
            floor = create_plane(color=TAN)

        robots = {
            "center": robot
        }
        return robots, return_order, return_order_lookup_dict
