

import numpy as np
import gym
import copy
import math
from controller import get_robot_qpos_from_obs
from policies.bucket_move import BucketPolicy
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


# Hugging Bucket
class NonPrehensileBucketPolicy(BucketPolicy):
    def __init__(self, env_name):
        super().__init__(env_name)

        ##### Replace with your code
        env = gym.make(env_name)
        self.action_space = copy.copy(env.action_space)
        env.close()
        del env

        self.obs_mode = 'pointcloud' # remember to set this!

        ### user policy
        self.left_arm_position_controller = self.init_position_controller(velocity_limit=[-20, 20], kp = 10, ki = 20, kd = 0)
        self.right_arm_position_controller = self.init_position_controller(velocity_limit=[-20, 20], kp = 10, ki = 20, kd = 0)
        self.position_controller = self.init_position_controller(velocity_limit = [-2000, 2000], kp = 10, ki = 1, kd = 0)
        self.height_position_controller = self.init_position_controller(velocity_limit = [-2000, 2000], kp = 10, ki = 1, kd = 0)
        self.rot_position_controller = self.init_position_controller(velocity_limit = [-2000, 2000], kp = 0.5, ki = 0.2, kd = 0)
        self.reset()

    def reset(self):
        self.grasps = None
        self.phase = 0
        self.step = 0
        self.count = 0
        self.last_q = []
        """ 
        0: 'move to pregrasp'
        1: 'move to grasp/contact'
        2: 'close gripper'
        3: 'pull back to pregrasp'
        """


    def act(self, obs):


        # Compose the velocities from all the different controllers
        current_q = get_robot_qpos_from_obs(obs, n_arms=2)
        forward_vel = 0
        theta_vel = 2
        height = 3
        right_arm_base = 4
        left_arm_base = 13
        right_arm_c_joint_1 = right_arm_base+1
        left_arm_c_joint_1 = left_arm_base+1

        right_arm_c_joint_2 = right_arm_base+3
        left_arm_c_joint_2 = left_arm_base+3

        right_arm_c_joint_3 = right_arm_base+5
        left_arm_c_joint_3 = left_arm_base+5

        c = 0.2
        action = np.zeros(22)
        if(self.phase == -1):
            self.phase = 0
        if(self.phase == 0):
            action[height] = -1
            action[forward_vel] = 1
        elif(self.phase == 1):    
            action[right_arm_c_joint_1] = c
            action[left_arm_c_joint_1] = c
            action[right_arm_c_joint_2] = -c
            action[left_arm_c_joint_2] = -c
            action[right_arm_c_joint_3] = -c
            action[left_arm_c_joint_3] = -c
            action[right_arm_base] = -c/2
            action[left_arm_base] = c/2
        elif(self.phase == 2):
            action[right_arm_c_joint_1] = c
            action[left_arm_c_joint_1] = c
            action[right_arm_c_joint_2] = -c
            action[left_arm_c_joint_2] = -c
            action[right_arm_c_joint_3] = -c
            action[left_arm_c_joint_3] = -c
            target_height = 0.35
            action[right_arm_base] = -c/2
            action[left_arm_base] = c/2

            desired_world_pose = [0,0]
            theta = current_q[2]
            rotation = np.array([[math.cos(theta),-math.sin(theta)],
                                [math.sin(theta), math.cos(theta)]])
            transformed = np.linalg.inv(rotation).dot(np.array([desired_world_pose[0]-current_q[0],
                                                                desired_world_pose[1]-current_q[1]]))

            target_vel = self.position_controller.control(np.array([0,0]), np.array(transformed))
            target_rotvel = self.rot_position_controller.control(np.array(current_q[2:3]), np.array([self.get_desired_theta(current_q)]), spherical=[0])
            target_height_vel = self.height_position_controller.control(current_q[3:4], [target_height])


            action[:2] = target_vel[:2]
            action[2] = target_rotvel[0]
            action[height] = target_height_vel[0]

        elif(self.phase == 3):
            action[right_arm_c_joint_1] = -c
            action[left_arm_c_joint_1] = -c
            action[right_arm_c_joint_2] = c
            action[left_arm_c_joint_2] = c
            action[right_arm_c_joint_3] = c
            action[left_arm_c_joint_3] = c
            action[right_arm_base] = c/2
            action[left_arm_base] = -c/2


        if(self.phase == 0 and self.count > 30):
            self.phase = 1
        elif(self.phase == 1 and self.count > 60):
            self.phase = 2
        elif(self.phase == 2 and self.count > 120):
            self.phase = 3

        self.count+=1

        return action

