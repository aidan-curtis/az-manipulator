
import numpy as np
import gym
import copy
import math
from controller import get_robot_qpos_from_obs
from policies.policy import DualAPolicy
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


def get_center(obs):
    from sklearn.decomposition import PCA

    chair_pc = []
    red_max_thresh = 0.7
    min_chair_height = 0.08
    max_chair_height = 2

    chair_red, = np.where(obs['pointcloud']['seg'][:, 0] == 0)
    pc = obs['pointcloud']['xyz'][chair_red, :]

    chair_max_height = np.max(pc, axis=0)[2]
    dense_idx, = np.where(obs['pointcloud']['xyz'][:, 2] > chair_max_height * 0.7)
    dense_idx = list(set(chair_red).intersection(set(dense_idx)))
    dense_pc = obs['pointcloud']['xyz'][dense_idx]
    dense_xy = dense_pc[:, :2]
    xy = np.mean(dense_xy, axis=0)

    def find_center(pc_xy, mean_xy, v, name):  ## yellow
        m = (pc_xy - mean_xy) @ np.outer(v, v) + mean_xy
        mmax = np.argmax(m, axis=0)[0]
        mmin = np.argmin(m, axis=0)[0]
        pmax = m[mmax]
        pmin = m[mmin]

        xy = np.mean([pmax, pmin], axis=0)
        return xy, mmax, mmin

    pca = PCA()
    pca.fit(dense_xy)
    if len(pca.components_) == 1:
        return None
    v1, v2 = pca.components_

    xy, mmax, mmin = find_center(dense_xy, xy, v1, 'main axis')

    # redo PCA
    pca = PCA()
    pca.fit(dense_xy - xy)
    v1, v2 = pca.components_

    chair_above_idx, = np.where(pc[:, 2] > chair_max_height * 0.2)
    chair_above_pc = pc[chair_above_idx]

    chair_above_xy = chair_above_pc[:, :2]
    chair_xy, mmax, mmin = find_center(chair_above_xy, xy, v2, 'chair')

    chair_z = [chair_above_pc[n] for n in [mmax, mmin]]
    # chair_z = [n for n in chair_z if n[2] < 3/4*chair_max_height][0][2]
    #
    # chair_below_idx, = np.where(chair_above_pc[:, 2] < chair_z * 1.2)
    # chair_below_pc = chair_above_pc[chair_below_idx]
    # chair_below_xy = chair_below_pc[:, :2]
    # chair_xy, mmax, mmin = find_center(chair_below_xy, xy, v2, 'chair below')

    # chair_back_xy = [n for n in chair_z if n[2] > 3/4*chair_max_height][0][:2]

    return chair_xy[0], chair_xy[1], chair_xy

def get_saved_chair_traj(visualize_filename):
    with open('temp/{}'.format(visualize_filename), 'rb') as handle:
        obs = pickle.load(handle)
    return get_chair_traj(obs)

# Chair
class ChairNudgePolicy(DualAPolicy):
    def __init__(self, env_name):
        super().__init__()

        ##### Replace with your code
        env = gym.make(env_name)
        self.action_space = copy.copy(env.action_space)
        env.close()
        del env

        self.obs_mode = 'pointcloud' # remember to set this!

        self.position_controller = self.init_position_controller(velocity_limit=[-2000, 2000], kp = 20, ki = 0.5, kd = 0.5)
        self.rot_position_controller = self.init_position_controller(velocity_limit=[-2000, 2000], kp = 0.5, ki = 0.2, kd = 0)
        self.arm_controller = self.init_position_controller(velocity_limit=[-0.5, 0.5], kp = 20, ki = 0.5, kd = 0.5)
        self.min_chair_height = 0.08
        self.max_chair_height = 2

        self.reset()

    def reset(self):
        self.count=0
        self.phase=0
        self.target_q = None
        self.last_q = [0] * 22

    def act(self, obs):
        try:
            self.count+=1
            current_q = get_robot_qpos_from_obs(obs, n_arms=2)
            change_xy = np.linalg.norm(current_q[:2] - self.last_q[:2])
            # if self.phase >= 1 and change_xy < 0.01:
            #     self.phase = 3

            if(self.target_q is None):
                #try:
                self.target_q = self.get_chair_traj(obs)
                #except:
                #    raise NotImplementedError

            #try:
            center = get_center(obs)
            #except: 
            #    raise EOFError

            #try:
            if center is not None:
                chairx, chairy, chair_back_xy = center
            else:
                chairx, chairy = self.chair_xy_last
            #except:
            #    raise MemoryError

         
            # dist_fr_back = np.linalg.norm([chair_back_xy - current_q[:2]])
            # print('back', chair_back_xy, 'dist from back', dist_fr_back)
            if np.linalg.norm([chairx, chairy]) <= 0.01:
                print('close to target')
                self.phase = 4


            action = np.zeros(22)
            if(self.phase == 0):

                new_target_q = copy.deepcopy(self.target_q)
                desired_world_pose = self.target_q[:2]
                theta = current_q[2]
                rotation = np.array([[math.cos(theta), -math.sin(theta)],
                                    [math.sin(theta), math.cos(theta)]])
                transformed = np.linalg.inv(rotation).dot(np.array([desired_world_pose[0]-current_q[0],
                                                                    desired_world_pose[1]-current_q[1]]))
                new_target_q[0], new_target_q[1] = transformed[0], transformed[1]
                target_vel = self.position_controller.control(np.array([0,0]), np.array(new_target_q[:2]))
                target_height_vel = self.position_controller.control(np.array(current_q[3:4]), np.array(new_target_q[3:4]))
                target_rotvel = self.rot_position_controller.control(np.array(current_q[2:3]), np.array(new_target_q[2:3]), spherical=[0])

                # target_arm =
                # target_arm_vel = self.arm_controller(current_q[4:])


                if(self.count>500):
                    target_vel = np.zeros(target_vel.shape)
                action[:2] = target_vel[:2]
                action[2] = target_rotvel[0]
                action[3] = target_height_vel[0]
                if(self.count > 50):
                    self.phase = 1

            elif(self.phase == 1):
                action = np.zeros(22)
                action[0]=10
                own_dist = math.sqrt(current_q[0]**2+current_q[1]**2)
                if(own_dist<0.6):
                    self.phase = 3

                c = 0.08
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

                action[right_arm_c_joint_1] = c
                action[left_arm_c_joint_1] = c
                action[right_arm_c_joint_2] = -c
                action[left_arm_c_joint_2] = -c
                action[right_arm_c_joint_3] = -c
                action[left_arm_c_joint_3] = -c
                action[right_arm_base] = -c/2
                action[left_arm_base] = c/2

            elif self.phase == 3 and self.count < 185:
                # chairx, chairy, _ = get_center(obs)

                target_q = copy.deepcopy(current_q)
                target_q[0] -= chairx
                target_q[1] -= chairy

                theta = current_q[2]
                rotation = np.array([[math.cos(theta), -math.sin(theta)],
                                     [math.sin(theta), math.cos(theta)]])
                transformed = np.linalg.inv(rotation).dot(np.array([target_q[0] - current_q[0],
                                                                    target_q[1] - current_q[1]]))
                target_q[:2] = transformed


                self.position_controller = self.init_position_controller(velocity_limit=[-1, 1], kp=10, ki=0.5, kd=0)
                target_vel = self.position_controller.control(np.asarray([0,0]), target_q[:2])
                action[:2] = target_vel[:2]  ## [::-1]

                # print('target_xy', target_q[:2])
                # print('current_xy', current_q[:2])
                # print('v_xy', action[:2])

                c = 0.08
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

                action[right_arm_c_joint_1] = c
                action[left_arm_c_joint_1] = c
                action[right_arm_c_joint_2] = -c
                action[left_arm_c_joint_2] = -c
                action[right_arm_c_joint_3] = -c
                action[left_arm_c_joint_3] = -c
                action[right_arm_base] = -c/2
                action[left_arm_base] = c/2

            # print(f'step {self.count}  | phase {self.phase}  |  dist to target {chairx, chairy}:  {np.linalg.norm([chairx, chairy])}')
            self.last_q = current_q
            self.chair_xy_last = [chairx, chairy]
       

            return action
        except:
            return np.zeros(22)


    def is_red(self, color):
        red_max_thresh = 0.7
        red_min_thresh = 0.05
        return  color[0]>red_max_thresh and \
                color[1]<red_min_thresh and \
                color[2]<red_min_thresh
    def get_pc(self, obs):
        pc = []
        pc_color = []
        chair_pc = []
        chair_pc_color = []

        for i in range(obs['pointcloud']['seg'].shape[0]):
            # Handle segmentation
            if(obs['pointcloud']['seg'][i, 0] == 0 and \
                obs['pointcloud']['xyz'][i, 2] > self.min_chair_height and \
                obs['pointcloud']['xyz'][i, 2] < self.max_chair_height and \
                not self.is_red(obs['pointcloud']['rgb'][i]) ):
                chair_pc.append(obs['pointcloud']['xyz'][i])
                chair_pc_color.append(obs['pointcloud']['rgb'][i])
            # Filter Floor
            if(obs['pointcloud']['xyz'][i, 2] > 0.1):
                pc.append(obs['pointcloud']['xyz'][i])
                pc_color.append(obs['pointcloud']['rgb'][i])

        pc = np.array(pc)
        pc_color = np.array(pc_color)
        chair_pc = np.array(chair_pc)
        chair_pc_color = np.array(chair_pc_color)
        return pc, chair_pc, pc_color, chair_pc_color

    def quick_get_pc(self, obs):
        chair_pc = []
        red_max_thresh = 0.7
        chair_red, = np.where(obs['pointcloud']['seg'][:, 0] == 0)
        chair_above, = np.where(obs['pointcloud']['xyz'][:, 2] > self.min_chair_height)
        chair_below, = np.where(obs['pointcloud']['xyz'][:, 2] < self.max_chair_height)
        chair_not_red, = np.where( (obs['pointcloud']['rgb'][:, 0]<red_max_thresh))
        chair_idx = list(set(chair_red).intersection(set(chair_above)).intersection(chair_not_red).intersection(chair_below))
        return obs['pointcloud']['xyz'][chair_idx, :]

    def get_chair_traj(self, obs):

        robots, return_order, return_order_lookup_dict = self.get_pb_robot()
        robot = robots["center"]
        robot_joints = get_joints(robot)
        joint_names = get_joint_names(robot, robot_joints)
        joint_lookup_dict = {name: joint for name, joint in zip(joint_names, robot_joints)}
        joint_position_vals = obs['agent']

        pb_joint_indices = [joint_lookup_dict[joint_name] for joint_name in return_order]
        pb_joint_values = [joint_position_vals[return_order_lookup_dict[joint_name]] for joint_name in return_order]

        set_joint_positions(robot, pb_joint_indices, pb_joint_values)

        # chairx, chairy = self.get_chair_xy(obs)

        # chairx, chairy, chair_back_xy = get_center(obs)

        center = get_center(obs)
        if center != None:
            chairx, chairy, chair_back_xy = center
        else:
            chairx, chairy = self.chair_xy_last

        # Move the base to align with the goal/chair vector. Need to transform world->base frame
        orig_joint_values = copy.deepcopy(pb_joint_values)
        approach_dist = 1.8  ## 2
        pb_joint_values[0], pb_joint_values[1] = chairx*approach_dist, chairy*approach_dist # In absolute coordinates -- to be converted later
        pb_joint_values[2] = math.atan(chairy/chairx)
        if(chairx*approach_dist<0):
            pb_joint_values[2] = pb_joint_values[2]+math.pi

        # Flip directions
        pb_joint_values[2]+=math.pi
        pb_joint_values[2]=pb_joint_values[2]%(2*math.pi)


        # Robot needs to crouch
        pb_joint_values[3]=0.3
        disconnect()
        # print(pb_joint_values)
        return pb_joint_values

