
import numpy as np
import gym
import copy
from pointcloud_utils import dist3d
from controller import get_robot_qpos_from_obs, open_gripper, equal, \
    closed_gripper, open_gripper
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
from controller import get_robot_qpos_from_obs, open_gripper, equal, \
    closed_gripper, open_gripper, dual_drake_ik
import open3d as o3d


# Prehensile Bucket
class PrehensileBucketPolicy(BucketPolicy):
    def __init__(self, env_name):
        super().__init__(env_name)

        ##### Replace with your code
        env = gym.make(env_name)
        self.action_space = copy.copy(env.action_space)
        env.close()
        del env

        self.obs_mode = 'pointcloud' # remember to set this!

        ### user policy
        self.reset()


    def get_target_qpos(self, phase, arm_q):
        if (phase == 0):  ## move to pregrasp
            target_gripper = open_gripper
        elif (phase == 1):  ## move to grasp
            target_gripper = open_gripper
        elif (phase == 2):  ## close gripper
            target_gripper = closed_gripper
        elif (phase == 5):
            target_gripper = open_gripper
        else: ## (phase == 3):  ## pull to pregrasp
            target_gripper = closed_gripper
        return arm_q, target_gripper

    def reset(self):

        self.left_arm_position_controller = self.init_position_controller(velocity_limit=[-1, 1], kp = 10, ki = 2, kd = 0)
        self.right_arm_position_controller = self.init_position_controller(velocity_limit=[-1, 1], kp = 10, ki = 2, kd = 0)
        self.position_controller = self.init_position_controller(velocity_limit = [-1, 1], kp = 20, ki = 0.5, kd = 0)
        self.height_position_controller = self.init_position_controller(velocity_limit = [-1, 1], kp = 10, ki = 1, kd = 0)
        self.rot_position_controller = self.init_position_controller(velocity_limit = [-1, 1], kp = 0.5, ki = 0.2, kd = 0)

        self.grasps = None
        self.phase = -1
        self.step = 0
        self.last_q = []
        self.counter = 0
        """ 
        0: 'move to pregrasp'
        1: 'move to grasp/contact'
        2: 'close gripper'
        3: 'pull back to pregrasp'
        """

    def get_bucket_conf_data(self, grasp_pose, q0):
        grasp_q = dual_drake_ik(grasp_pose[1][0], grasp_pose[1][1], grasp_pose[0][0], grasp_pose[0][1], q0)
        grasp_q[3]+=0.2
        return grasp_q

    def parse_grasps(self, grasps):
        grasp_pairs = []
        for left_grasp, left_dist, _ in grasps:
            for right_grasp, _, right_dist in grasps:
                pair_distance = dist3d(left_grasp[0], right_grasp[0])
                if(pair_distance>0.2):
                    grasp_pairs.append( (left_grasp, right_grasp, left_dist**2+right_dist**2) )

        return sorted(grasp_pairs, key=lambda x: x[2])

    def act(self, obs):
        try:
            ### get grasps generated based on point cloud when scene changed
            current_q = get_robot_qpos_from_obs(obs, n_arms=2)

            if self.grasps == None:
                _, rim_pc, _, _ = self.get_pc(obs)

                self.bucket_x, self.bucket_y = np.mean(np.array(rim_pc[:, 0]), axis=0), np.mean(np.array(rim_pc[:, 1]), axis=0)

                self.grasps = self.get_grasps(obs, num=20, only_pose=True, downsample=5)
                self.grasp_pairs = self.parse_grasps(self.grasps)
                self.grasp_conf = self.get_bucket_conf_data(self.grasp_pairs[0], current_q)

            target_q, target_gripper = self.get_target_qpos(self.phase, self.grasp_conf)

            new_target_q = copy.deepcopy(target_q)
            desired_world_pose = target_q[:2]
            theta = current_q[2]
            rotation = np.array([[math.cos(theta),-math.sin(theta)],
                                [math.sin(theta), math.cos(theta)]])
            transformed = np.linalg.inv(rotation).dot(np.array([desired_world_pose[0]-current_q[0],
                                                                desired_world_pose[1]-current_q[1]]))

            goto = 70
            if(self.phase == -1):
                if(self.counter >= 10):
                    self.phase = 0
            if(self.phase == 0):
                if(self.counter >= goto):
                    self.phase = 1
            elif(self.phase == 1):
                if(self.counter >= goto+15):
                    self.phase = 2
                new_target_q[3]-=2
            elif(self.phase == 2):
                if(self.counter >= goto+25):
                    self.phase = 3
                new_target_q[3]-=2
            elif(self.phase == 3):
                if(self.counter >= goto+40):
                    self.start_q = current_q
                    self.phase = 4
            elif(self.phase == 5):
                new_target_q[3]+=0.2

            if(self.phase == 4):
                desired_world_pose = [self.start_q[0]-self.bucket_x, self.start_q[1]-self.bucket_y]
                theta = current_q[2]
                rotation = np.array([[math.cos(theta),-math.sin(theta)],
                                    [math.sin(theta), math.cos(theta)]])
                transformed = np.linalg.inv(rotation).dot(np.array([desired_world_pose[0]-current_q[0],
                                                                    desired_world_pose[1]-current_q[1]]))

                new_target_q[:2] = transformed
                target_vel = self.position_controller.control(np.array([0,0]), np.array(transformed))
                # target_rotvel = self.rot_position_controller.control(np.array(current_q[2:3]), np.array([self.get_desired_theta(current_q)]), spherical=[0])

                if(self.counter >= 160):
                    self.phase = 5
            else:
                new_target_q[0], new_target_q[1] =  transformed[0], transformed[1]
                target_vel = self.position_controller.control(np.array([0,0]), np.array(new_target_q[:2]))

            target_rotvel = self.rot_position_controller.control(np.array(current_q[2:3]), np.array(new_target_q[2:3]), spherical=[0])
            target_height_vel = self.height_position_controller.control(np.array(current_q[3:4]), np.array(new_target_q[3:4]))
            target_right_arm_vel = self.right_arm_position_controller.control(np.array(current_q[4:11]), np.array(target_q[4:11]))
            target_left_arm_vel = self.left_arm_position_controller.control(np.array(current_q[13:20]), np.array(target_q[13:20]))

            action = np.zeros(22)
            if(self.phase < 5):    
                action[:2] = target_vel[:2]
                action[2] = target_rotvel[0]
                if(self.phase>=0):
                    action[4:11] = target_right_arm_vel
                    action[13:20] = target_left_arm_vel
                
            action[3] = target_height_vel[0]
            action[20:22] = target_gripper
            action[11:13] = target_gripper

            right_diff = np.linalg.norm(current_q[4:11] - target_q[4:11])
            left_diff = np.linalg.norm(current_q[13:20] - target_q[13:20])
            base_diff = np.linalg.norm(current_q[:2] - new_target_q[:2])

            print(f'step {self.counter}  | phase {self.phase}, diff in left q {left_diff}  diff in right q {right_diff} diff in base {base_diff}')

            self.counter+=1
        except:
            action = np.zeros(22)
        return action

        # 43 big rim 212
