import numpy as np
import gym
import copy
from controller import get_robot_qpos_from_obs, open_gripper, equal, \
    closed_gripper, open_gripper, drake_ik
from policies.policy import SingleAPolicy
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
from pointcloud_utils import dist
import pybullet as p
from sklearn.cluster import DBSCAN

import zmq
import zlib
import pickle5
import threading
import functools
import pickle


# context = zmq.Context()
# socket = context.socket(zmq.REQ)
# socket.connect("tcp://3.17.37.135:5555")

# Drawer
class DrawerPolicy(SingleAPolicy):
    def __init__(self, env_name):
        super().__init__()

        ##### Replace with your code
        env = gym.make(env_name)
        self.action_space = copy.copy(env.action_space)
        env.close()
        del env

        self.obs_mode = 'pointcloud' # remember to set this!

        ### user policy
        self.reset()

    def get_pc(self, obs):
        pc = []
        pc_color = []
        grasp_pc = []
        grasp_pc_color = []

        for i in range(obs['pointcloud']['seg'].shape[0]):
            # Handle segmentation
            if(obs['pointcloud']['seg'][i, 0]):
                grasp_pc.append(obs['pointcloud']['xyz'][i])
                grasp_pc_color.append(obs['pointcloud']['rgb'][i])
            # Filter Floor
            if(obs['pointcloud']['xyz'][i, 2] > 0.1):
                pc.append(obs['pointcloud']['xyz'][i])
                pc_color.append(obs['pointcloud']['rgb'][i])

        pc = np.array(pc)
        pc_color = np.array(pc_color)
        grasp_pc = np.array(grasp_pc)
        return pc, grasp_pc, pc_color, grasp_pc_color

    def get_conf_data(self, grasp_pose, robots, name_link_dict, q0):
        approach_grasp = Pose(point=Point(z = -0.1))
        grasp_trans = Pose(point=Point(z = 0.04))
        post_grasp = Pose(point=Point(z = -0.5))
        real_grasp_pose = multiply(grasp_pose, grasp_trans)
        approach_grasp_pose = multiply(real_grasp_pose, approach_grasp)
        post_grasp_pose = multiply(real_grasp_pose, post_grasp)

        message = {"message_name": "drake_ik",
                   "xyz":real_grasp_pose[0], 
                   "quat":real_grasp_pose[1], 
                   "q0":list(q0), 
                   "ARM_ONLY":False, 
                   "verbose":True,
                   "hand_link":"right_panda_link8", 
                   "dimensions":[], 
                   "center":[]}

        socket.send(zlib.compress(pickle.dumps(message)))
        result = pickle.loads(zlib.decompress(socket.recv()))
        grasp_q = list(result["grasp_q"])

        grasp_q[3]-=0.02 # We can only see the top of handles
        pre_q = list(grasp_q)
        pre_q[0]-=0.1

        post_q = list(grasp_q)
        post_q[0]-=1

        return (pre_q, grasp_q, post_q)


    def generate_grasps(self, grasp_pc):
        grasp_poses = []

        db = DBSCAN(eps=0.1, min_samples=3).fit(grasp_pc)
        labels = list(db.labels_) 
        if(len(labels)>0):
            label_mode = max(set(labels), key=labels.count)
            valid_idx,  = np.where(labels==label_mode)
            mean_point = np.mean(grasp_pc[valid_idx, :], axis=0)
        else:
            mean_point = np.mean(grasp_pc[:, :], axis=0)

            
        for pt in grasp_pc:
            r = R.from_matrix([[1,  0, 0],
                               [0,  0, 1],
                               [0, -1, 0]])
            quat = list(r.as_quat())
            score = [0]
            pos = [pt[0]-0.02, pt[1], pt[2]]
            grasp_poses.append(list(pos)+quat+score)


        grasp_poses.sort(key=lambda s: dist(s[:3], mean_point))
        return grasp_poses


    def has_cabinet_moved(self, obs):
        # Get current cabinet points
        _, grasp_pc, _, _ = self.get_pc(obs)
        initial_mean = np.mean(self.initial_grasp_pc, axis=0)
        closest_dist = []
        for point in grasp_pc:
            min_dist = float("inf")
            for initial_point in self.initial_grasp_pc:
                if(np.linalg.norm(point-initial_point)<min_dist):
                    min_dist = np.linalg.norm(point-initial_point)
            closest_dist.append(min_dist)

        # print(np.linalg.norm(closest_dist))
        return np.linalg.norm(closest_dist)>0.08


    def update_phase(self, phase, grasp, current_q, obs):
        pre_q, grasp_q, post_q = grasp
        if phase == 0 and (equal(current_q[:11], pre_q[:11], epsilon = 0.08) and equal(current_q, self.last_q, epsilon = 0.0003)):
            phase = 1
        if phase == 1 and (equal(current_q[:11], grasp_q[:11], epsilon = 0.08) and equal(current_q, self.last_q, epsilon = 0.003)):
            phase = 2
        elif phase == 2:
            if equal(current_q, self.last_q, epsilon = 0.001):
                phase = 3
        elif phase == 3 and equal(current_q[:11], post_q[:11], epsilon = 0.2):
            phase = -1

        elif phase == 3 and self.phase3_counter == 20 and not self.has_cabinet_moved(obs):
            # Check if the cabinet has moved
            phase = -1

        if(phase == 3):
            self.phase3_counter+=1
        else:
            self.phase3_counter=0

        self.last_q = current_q
        return phase

    def get_target_qpos(self, phase, grasp):
        pre_q, grasp_q, post_q = grasp
        if (phase == 0):  ## move to pregrasp
            target_q = pre_q
            target_gripper = open_gripper
        elif (phase == 1):  ## move to grasp
            target_q = grasp_q
            target_gripper = open_gripper
        elif (phase == 2):  ## close gripper
            target_q = grasp_q
            target_gripper = closed_gripper
        else: ## (phase == 3):  ## pull to pregrasp
            target_q = post_q
            target_gripper = closed_gripper
        return target_q, target_gripper

    def reset(self):
        self.grasps = None
        self.phase = 0
        self.phase3_counter = 0 
        self.step = 0
        self.last_q = []

        self.base_controller = self.init_position_controller(velocity_limit=[-20, 20], kp = 20, ki = 0.5, kd = 0)
        self.slow_base_controller = self.init_position_controller(velocity_limit=[-0.3, 0.3], kp = 20, ki = 0.5, kd = 0)
        self.med_base_controller = self.init_position_controller(velocity_limit=[-0.5, 0.5], kp = 20, ki = 0.5, kd = 0)
        self.arm_controller = self.init_position_controller(velocity_limit=[-0.5, 0.5], kp = 10, ki = 5, kd = 0)
        """ 
        0: 'move to pregrasp'
        1: 'move to grasp/contact'
        2: 'close gripper'
        3: 'pull back to pregrasp'
        """

    def act(self, obs):
        try:
            ### get grasps generated based on point cloud when scene changed
            if self.grasps == None:
                _, self.initial_grasp_pc, _, _ = self.get_pc(obs)
                self.grasps = self.get_grasps(obs, num=20)

            current_q = get_robot_qpos_from_obs(obs)

            ## detect phase changes and failed grasp
            self.phase = self.update_phase(self.phase, self.grasps[0], current_q, obs)
            if self.phase == -1:
                self.phase = 0

            target_q, target_gripper = self.get_target_qpos(self.phase, self.grasps[0])
            if(self.phase == 3):
                if(self.phase3_counter<=20):
                    target_base_vel = self.slow_base_controller.control(current_q[:3], target_q[:3])
                else:
                    target_base_vel = self.med_base_controller.control(current_q[:3], target_q[:3])
            else:
                target_base_vel = self.base_controller.control(current_q[:3], target_q[:3])

            target_arm_vel = self.arm_controller.control(current_q[3:], target_q[3:])
            target_vel = list(target_base_vel)+list(target_arm_vel)
            action = target_vel
            action[11:13] = target_gripper

            diff = np.linalg.norm(current_q[:11] - target_q[:11])
            #print(f'step {self.step}  | phase {self.phase}, diff in q {diff}')
            self.step += 1
        except:
            action = np.zeros(13)
        return action