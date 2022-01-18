import numpy as np
import gym
import copy
from controller import get_robot_qpos_from_obs, equal
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
from pointcloud_utils import dist, get_parent_from_tool
import pybullet as p
from pprint import pprint
from controller import get_robot_qpos_from_obs, open_gripper, equal, \
    closed_gripper, open_gripper, drake_ik


open_gripper = [0.0002, 0.0002]  ## qvel[-2:]
fast_open_gripper = [1, 1]  ## qvel[-2:]

closed_gripper = [-1, -1]  ## qvel[-2:]

## for openning the drawer
class DoorPolicy(SingleAPolicy):
    def __init__(self, env_name):
        super().__init__()

        ##### Replace with your code
        env = gym.make(env_name)
        self.action_space = copy.copy(env.action_space)
        env.close()
        del env

        self.obs_mode = 'pointcloud' # remember to set this!

        ### user policy
        self.straight_motion = True
        self.can_env_name = "OpenCabinetDoor"
        self.env_name = env_name
        self.phases = {}
        self.reset()

    def reset(self):
        """ phases
        0: 'move to pregrasp'
        1: 'move to grasp/contact'
        2: 'close gripper'
        3: 'pull back to pregrasp'
        """
        if len(self.phases) > 0:
            pprint(self.phases)
            print('\n\n')
        self.grasps = None
        self.phase = 0
        self.step = 0
        # self.position_controller = self.init_position_controller(velocity_limit=[-20, 20], kp = 10, ki = 20, kd = 0)
        self.reset_position_controllers()
        self.count = 0
        self.last_q = None
        self.phases = {}
        self.fixed_base = None
        self.fixed_arm = None
        self.xyt_desired = None
        self.desired_angle = -np.pi/2
        self.fixed_x_min = None

        ## estimate once at the begining
        self.estimated_door_width = None
        self.is_open_right = None
        self.est_x_axis, self.est_y_axis = None, None
        self.handle_horizontal = None
        self.handle_small = False
        self.maybe_open_left = False

        ## may need to check every step in some phases
        self.door_moved = None
        self.segmented_pc = None
        self.prepush_xy = None
        self.prepush_xy = None
        self.q_last = None
        self.a_last = None

        self.previous_wedge_x = 0

    def reset_position_controllers(self):
        self.base_controller = self.init_position_controller(velocity_limit=[-0.5, 0.5], kp = 20, ki = 0.5, kd = 0)
        self.arm_controller = self.init_position_controller(velocity_limit=[-0.5, 0.5], kp = 10, ki = 5, kd = 0)


    def update_phase(self, phase, grasp, current_q,
                     env_name="OpenCabinetDrawer-v0", obs=None):
        pre_q, grasp_q, post_q = grasp
        open_left = not self.is_open_right

        if(self.last_q is None):
            self.last_q = np.zeros(current_q.shape)
        def pregrasp_ready(alpha=0.005, beta=0.0003):
            if self.handle_small:
                (equal(current_q[:11], pre_q[:11], epsilon=0.005) and equal(current_q, self.last_q, epsilon = beta))
            return (equal(current_q[:11], pre_q[:11], epsilon = alpha) and equal(current_q[1], pre_q[1], epsilon = alpha/4)) or equal(current_q, self.last_q, epsilon = beta)##  and equal(current_q, last_q, epsilon = beta))

        def grasp_ready(alpha=0.9, beta=0.003, gamma=0.003):
            if self.handle_small:
                ((equal(current_q[:11], pre_q[:11], epsilon=0.02) and equal(current_q, self.last_q, epsilon = beta)))
            return (equal(current_q[:11], grasp_q[:11], epsilon = alpha) and equal(current_q, self.last_q, epsilon = beta)) or equal(current_q, self.last_q, epsilon = gamma)

        def grasp_stable():
            return equal(current_q, self.last_q, epsilon=0.001)

        def grasp_failed():
            return current_q[-1] < 0.002 and current_q[-2] < 0.002

        def door_opened_enough():
            door_moved = self.check_door_moved(obs=obs)
            return door_moved != None and (door_moved > self.estimated_door_width / 4) or door_moved > 0.15

        def push_no_more():
            if self.is_open_right: epsilon=0.05
            else: epsilon=0.02
            at_position = equal(current_q[0], self.get_prepush_xy(obs)[0], epsilon=epsilon)
            hitdoor = equal(current_q[0], self.get_prepush_xy(obs)[0], epsilon = 0.2) and equal(current_q, self.last_q, epsilon = 0.02)
            return at_position or hitdoor

        def get_stuck(beta=0.004):
            return equal(current_q, self.last_q, epsilon = beta)

        if phase == 0 and pregrasp_ready():
            phase = 1

        elif phase == 1 and grasp_ready(): ## alpha=0.2, beta=3
            phase = 2

        elif phase == 2:
            if grasp_failed() and self.check_away_from_handle(obs, epsilon=0.05):
                phase = -1
            elif grasp_stable():
                phase = 39

        elif phase == 3 and equal(current_q[:11], post_q[:11], epsilon = 0.2):
            phase = 31

        elif phase == 39:
            gripper_lost = grasp_failed()
            door_moved = self.check_door_moved(obs=obs)
            # print('door_moved', door_moved)
            # print('self.estimated_door_width/8', self.estimated_door_width/8)
            # print('gripper_lost', gripper_lost)
            # print('current_q[:2]', (current_q[-1], current_q[-2]))

            if door_opened_enough():
                phase = 41

            if self.check_away_from_handle(obs):
                if door_moved > self.estimated_door_width/8:
                    phase = 41
                else:
                    phase = -1

        elif phase == 41:

            if equal(current_q[0], self.get_far_x(obs), epsilon = 0.02):
                phase = 42
                self.fixed_base = None
            elif get_stuck():
                phase = 42
                self.fixed_base = None

        elif phase == 42 and equal(current_q[2], self.desired_angle, epsilon = 0.1):
            phase = 43  ##  * (-1) ** open_left
            self.fixed_base = None

        elif phase == 43 and equal(current_q[1], self.get_prepush_xy(obs)[1], epsilon=0.02):
            phase = 44
            self.fixed_base = None

        elif phase == 44 and push_no_more():
            phase = 45
            self.fixed_base = None

        self.last_q = current_q
        return phase

    def get_ranges(self, pc):
        return np.max(pc, axis=0) - np.min(pc, axis=0)

    def estimate_door_width(self, segmented_pc):
        handle_pc, door_pc, body_pc, robot_pc = segmented_pc

        x_range, y_range, z_range = self.get_ranges(door_pc)
        door_width = y_range ##np.sqrt(x_range**2 + y_range**2)

        x_range, y_range, z_range = self.get_ranges(body_pc)
        cabinet_width = y_range
        self.num_doors = round(cabinet_width / door_width)
        handle_horizontal = self.check_handle_horizontal(handle_pc)

        ## if the door has a single door and horizontal handle
        if self.num_doors == 1:
            self.maybe_open_left = True
            # print('\n\nthis door maybe opens left\n\n')

        ## two vertical handles, measure the distance between the edge of door and handle
        self.handle_gap = 100
        if not handle_horizontal:
            if self.is_open_right:
                y_door = np.max(door_pc, axis=0)[1]
                y_handle = np.max(handle_pc, axis=0)[1]
            else:
                y_door = np.min(door_pc, axis=0)[1]
                y_handle = np.min(handle_pc, axis=0)[1]
            self.handle_gap = np.abs(y_handle - y_door)

        return door_width

    def estimate_door_axis(self, segmented_pc, open_right):
        door_pc = segmented_pc[1]
        if open_right:  ## find the max of y direction
            y_axis = np.min(door_pc, axis=0)[1]
        else:
            y_axis = np.max(door_pc, axis=0)[1]
        x_axis = np.max(door_pc, axis=0)[0]
        # print('x_axis', x_axis)
        # print('y_axis', y_axis)
        return x_axis, y_axis

    def get_far_x(self, obs=None):
        if self.segmented_pc == None:
            _, self.segmented_pc, _, _ = self.get_pc(obs)
        door_pc = self.segmented_pc[1]
        x_min = np.min(door_pc, axis=0)[0]
        open_right = self.is_open_right
        if open_right:
            return x_min - 1.1 ## 1.25
        return x_min - 1 ##1

    def get_offset_xy(self, obs=None, x_off=0.0, y_off=0.0):
        if self.segmented_pc == None:
            _, self.segmented_pc, _, _ = self.get_pc(obs)
        door_pc = self.segmented_pc[1]
        x_min = np.min(door_pc, axis=0)[0]
        if self.is_open_right:
            y_max = np.max(door_pc, axis=0)[1]
        else:
            y_max = np.min(door_pc, axis=0)[1]
            y_off = - y_off
        return x_min + x_off, y_max + y_off

    def get_prepush_xy(self, obs=None):
        if self.prepush_xy == None:
            if self.handle_gap < 0.03:
                self.prepush_xy = self.get_offset_xy(obs=obs, x_off=-0.05, y_off=0.2)  ## -0.2
            if self.estimated_door_width > 0.3:
                x_off = max([0.03, self.check_door_moved(obs)-0.08])
                
                self.prepush_xy = self.get_offset_xy(obs=obs, x_off=x_off, y_off=0.2)  ## -0.2
            else:
                self.prepush_xy = self.get_offset_xy(obs=obs, x_off=-0.05, y_off=0.2)  ## -0.2
        return self.prepush_xy

    def get_postpush_xy(self, obs=None):
        if self.postpush_xy == None:
            self.postpush_xy = self.get_offset_xy(obs=obs, x_off=0.1, y_off=-2) ## -0.2
        return self.postpush_xy

    def check_door_moved(self, segmented_pc=None, obs=None):
        if self.door_moved == None:
            if segmented_pc == None:
                _, segmented_pc, _, _ = self.get_pc(obs)
                self.segmented_pc = segmented_pc
            door_pc = self.segmented_pc[1]
            x_min = np.min(door_pc, axis=0)[0]
            if self.fixed_x_min == None:
                self.fixed_x_min = x_min
            self.door_moved = abs(self.fixed_x_min - x_min)

            if self.door_moved > 0.02 and self.maybe_open_left:
                for xyz in door_pc:
                    if xyz[0] == x_min:
                        if abs(xyz[1] - self.est_y_axis) < 0.05:
                            # print('original open right', self.is_open_right)
                            self.is_open_right = not self.is_open_right
                            # print('updated open right', self.is_open_right)
                            # print('original door axis', self.est_x_axis, self.est_y_axis)
                            self.est_x_axis, self.est_y_axis = self.estimate_door_axis(segmented_pc, self.is_open_right)

                            # print('updated door axis', self.est_x_axis, self.est_y_axis)
                            self.maybe_open_left = False
                            break

            ## need to calculate again in some cases
            self.prepush_xy = None
            self.postpush_xy = None
        return self.door_moved

    def distance(self, pt1, pt2):
        return np.sqrt( (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 )

    def check_away_from_handle(self, obs, epsilon=0.2):
        """ the points near x_max on robot are away from
            the points near x_min on the handle """
        if self.segmented_pc == None:
            _, self.segmented_pc, _, _ = self.get_pc(obs)
        handle_pc, door_pc, body_pc, robot_pc = self.segmented_pc
        if len(handle_pc) == 0: ## no handle found
            return False
        xyz_mean_door = np.mean(handle_pc, axis=0)

        x_max_robot = np.max(robot_pc, axis=0)[0]
        for xyz in robot_pc:
            if x_max_robot == xyz[0]:
                if self.distance(xyz_mean_door[:2], xyz[:2]) > epsilon:
                    return True
        return False


    def check_open_right(self, segmented_pc):
        """ return True if the door opens to the right side """
        handle_pc, door_pc, body_pc, robot_pc = segmented_pc

        ## if the handle is horizontal, the door body
        ## should lie on the right half of the object body
        if self.check_handle_horizontal(handle_pc):
            if np.mean(door_pc, axis=0)[1] < np.mean(body_pc, axis=0)[1]:
                result = True
            else:
                result = False

        ## if the handle is vertical, the center of the handle on the y axis
        ## should be to the left of the center of the door on the y axis
        else:
            if np.mean(handle_pc, axis=0)[1] > np.mean(door_pc, axis=0)[1]:
                result = True
            else:
                result = False
        return result

    def check_handle_horizontal(self, grasp_pc):
        """ check the ratio of y_range / x_range """
        if self.handle_horizontal == None:
            x_range, y_range, z_range = self.get_ranges(grasp_pc)
            if x_range > z_range and y_range > z_range:
                handle_horizontal = True
            elif z_range > x_range and z_range > y_range:
                handle_horizontal = False
            else:
                handle_horizontal = True
            if y_range < 0.03 and z_range < 0.03:
                self.handle_small = True
                handle_horizontal = True
            self.handle_horizontal = handle_horizontal
            # print('check_handle_horizontal', self.handle_horizontal)
        return self.handle_horizontal

    def get_target_qpos(self, phase, grasp, env_name="OpenCabinetDrawer-v0", obs=None):
        pre_q, grasp_q, post_q = grasp
        target_gripper = open_gripper
        open_left = not self.is_open_right
        qpos = get_robot_qpos_from_obs(obs)

        if (phase == 0):  ## move to pregrasp
            target_q = pre_q
            target_gripper = open_gripper
        elif (phase == 1):  ## move to grasp
            target_q = grasp_q
            # print('    grasp x,y,z', grasp_q[:3])
            # print('    current x,y,z', qpos[:3])
            target_gripper = open_gripper
        elif (phase == 2):  ## close gripper
            target_q = grasp_q
            target_gripper = closed_gripper
        elif (phase == 3):  ## pull to postgrasp
            target_q = post_q
            target_gripper = closed_gripper

        elif phase == 31: ## for openning the door, stick the hand vertically at the gap
            target_q = pre_q
            target_gripper = closed_gripper
        elif phase == 32: ## push the door from the inside
            target_q = post_q
            target_gripper = closed_gripper

        elif phase == 39: ## for moving in a circular trajectory
            x_axis, y_axis = self.est_x_axis, self.est_y_axis
            # print('\n\n door moved', self.door_moved)
            qpos = get_robot_qpos_from_obs(obs)
            x_robot, y_robot, theta_robot = qpos[:3]

            def change_target():
                return equal(self.xyt_desired, qpos[:3], epsilon=0.1)

            # if isinstance(self.xyt_desired, tuple):
                # print('   diff in xyt', np.linalg.norm(np.asarray(self.xyt_desired) - qpos[:3]))

            open_left = not self.is_open_right


            desired_world_pose = [qpos[0]-2, qpos[1]-0.5 * (-1)**open_left]

            theta = qpos[2]
            rotation = np.array([[math.cos(theta),-math.sin(theta)],
                                [math.sin(theta), math.cos(theta)]])
            transformed = np.linalg.inv(rotation).dot(np.array([desired_world_pose[0]-qpos[0],
                                                                desired_world_pose[1]-qpos[1]]))
            target_q = copy.deepcopy(qpos)
            target_q[0], target_q[1] =  transformed[0], transformed[1]


            # desired_world_pose = [qpos[0]-1, qpos[1]-0.02 * (-1)**open_left]
            # target_q = copy.deepcopy(qpos)
            # target_q[0], target_q[1] =  desired_world_pose[0], desired_world_pose[1]

            if not isinstance(self.fixed_arm, np.ndarray):
                self.fixed_arm = target_q[3:]
            else:
                # print('   desired x, y, theta', target_q[:3])
                # print('   current x, y, theta', qpos[:3])
                # print('   desired arm', [round(n, 2) for n in self.fixed_arm])
                # print('   current arm', [round(n, 2) for n in qpos[3:]])

                target_q[3:] = self.fixed_arm
            target_gripper = closed_gripper

        elif phase == 41:  ## for moving back
            x_far = self.get_far_x(obs)
            # print('     current x', qpos[0])
            # print('     far x', x_far)
            target_q = copy.deepcopy(qpos)
            target_q[0] = x_far
            if not isinstance(self.fixed_base, np.ndarray):
                self.fixed_base = qpos[1:]
            else:
                target_q[1:] = self.fixed_base
            target_gripper = fast_open_gripper


        elif phase == 42:
            target_q = copy.deepcopy(qpos)
            if not isinstance(self.fixed_base, np.ndarray):
                self.fixed_base = qpos
            else:
                target_q = self.fixed_base
            target_q[2] = self.desired_angle ## * (-1) ** open_left
            target_gripper = fast_open_gripper

        elif phase == 43:
            target_q = copy.deepcopy(qpos)
            if not isinstance(self.fixed_base, np.ndarray):
                self.fixed_base = qpos
            else:
                target_q = self.fixed_base
            target_q[1] = self.get_prepush_xy(obs)[1]
            # print('     target x,y', target_q[:2])
            # print('     current x,y', qpos[:2])
            target_gripper = fast_open_gripper

        elif phase == 44:
            target_q = copy.deepcopy(qpos)
            if not isinstance(self.fixed_base, np.ndarray):
                self.fixed_base = qpos
            else:
                target_q = self.fixed_base
            target_q[:2] = self.get_prepush_xy(obs)
            # print('     target x,y', target_q[:2])
            # print('     current x,y', qpos[:2])
            target_gripper = fast_open_gripper

        elif phase == 45:
            target_q = copy.deepcopy(qpos)
            if not isinstance(self.fixed_base, np.ndarray):
                self.fixed_base = qpos
            else:
                target_q = self.fixed_base
            target_q[:2] = self.get_postpush_xy(obs)
            # print('     target x,y', target_q[:2])
            # print('     current x,y', qpos[:2])
            target_gripper = fast_open_gripper

        return target_q, target_gripper

    def generate_grasps(self, segmented_pc):
        grasp_poses = []
        grasp_pc, door_pc, body_pc, robot_pc = segmented_pc

        mean_point = np.mean(grasp_pc, axis=0)
        horizontal = self.check_handle_horizontal(grasp_pc)  ## some handles orient vertically, some horizontally
        for pt in grasp_pc:
            score = [0]
            pos = [pt[0]-0.02, pt[1], pt[2]]

            if horizontal:
                r = R.from_matrix([[1,  0, 0],
                                   [0,  0, 1],
                                   [0, -1, 0]])
            else:
                r = R.from_matrix([[1,  0, 0],
                                   [0,  1, 0],
                                   [0,  0, 1]])
            open_right = self.is_open_right
            if open_right: ## and self.handle_horizontal:
                if self.handle_gap > 0.03 and not self.handle_small:
                    pos[1] -= 0.05
            if self.estimated_door_width > 0.3:
                pos[0] -= 0.05
            if not self.handle_horizontal:
                pos[2] = mean_point[2]

            quat = list(r.as_quat())
            grasp_poses.append(list(pos)+quat+score)

        grasp_poses.sort(key=lambda s: dist(s[:3], mean_point))
        return grasp_poses


    def get_conf_data(self, grasp_pose, robots, name_link_dict, q0):
        approach_grasp = Pose(point=Point(z = -0.1))
        grasp_trans = Pose(point=Point(z = 0.04))
        post_grasp = Pose(point=Point(z = -0.5))
        real_grasp_pose = multiply(grasp_pose, grasp_trans)
        approach_grasp_pose = multiply(real_grasp_pose, approach_grasp)
        post_grasp_pose = multiply(real_grasp_pose, post_grasp)
        # grasp_q = drake_ik(real_grasp_pose[0], real_grasp_pose[1], list(q0))
        grasp_q = p.calculateInverseKinematics(robots['center'], name_link_dict["right_panda_link8"],
                                               real_grasp_pose[0],
                                               real_grasp_pose[1])

        grasp_q = list(grasp_q)
        pre_q = list(grasp_q)
        pre_q[0] -= 0.1
        post_q = list(grasp_q)
        post_q[0] -= 0.5 * self.estimated_door_width
        # post_q[1] -= 0.25 * self.estimated_door_width * (-1) ** (not self.is_open_right)

        grasp_q[0] +=  0.1
        return (pre_q, grasp_q, post_q)

    def get_pc(self, obs):

        pc = []
        pc_color = []
        grasp_pc = []
        door_pc = []
        body_pc = []
        robot_pc = []

        for i in range(obs['pointcloud']['seg'].shape[0]):
            # Handle segmentation
            if(obs['pointcloud']['seg'][i, 0]):
                grasp_pc.append(obs['pointcloud']['xyz'][i])
                pc.append(obs['pointcloud']['xyz'][i])
                pc_color.append([1, 0, 0])

            # Door segmentation
            if(obs['pointcloud']['seg'][i, 1] and not obs['pointcloud']['seg'][i, 0]):
                door_pc.append(obs['pointcloud']['xyz'][i])
                pc.append(obs['pointcloud']['xyz'][i])
                pc_color.append([0, 1, 0])

            # Filter Floor
            if(obs['pointcloud']['xyz'][i, 2] > 0.1):


                # filter out the robot
                if not obs['pointcloud']['seg'][i, 2]:
                    body_pc.append(obs['pointcloud']['xyz'][i])

                    if not obs['pointcloud']['seg'][i, 0] and not obs['pointcloud']['seg'][i, 1]:
                        pc.append(obs['pointcloud']['xyz'][i])
                        pc_color.append([0, 0, 1])
                else:
                    robot_pc.append(obs['pointcloud']['xyz'][i])

        grasp_pc_arr = np.array(grasp_pc)
        door_pc_arr = np.array(door_pc)
        body_pc_arr = np.array(body_pc)
        robot_pc_arr = np.array(robot_pc)
        segmented_pc = [grasp_pc_arr, door_pc_arr, body_pc_arr, robot_pc_arr]

        # TODO: Move these to a better place
        if self.is_open_right == None or self.est_x_axis == None:
            self.is_open_right = self.check_open_right(segmented_pc)
            self.est_x_axis, self.est_y_axis = self.estimate_door_axis(segmented_pc, self.is_open_right)

        for xyz in door_pc:
            if abs(xyz[0] - self.est_x_axis) <= 0.02 and abs(xyz[1] - self.est_y_axis) <= 0.02:
                pc.append(xyz)
                pc_color.append([0, 0, 0])

        pc = np.array(pc)
        pc_color = np.array(pc_color)
        return pc, segmented_pc, pc_color, None

    def act(self, obs):

        ### get grasps generated based on point cloud when scene changed
        if self.grasps == None:
            _, segmented_pc, _, _ = self.get_pc(obs)
            self.estimated_door_width = self.estimate_door_width(segmented_pc)
            # print('self.estimated_door_width', self.estimated_door_width)
            self.grasps = self.get_grasps(obs, num=20)
            self.start_obs = obs

        current_q = get_robot_qpos_from_obs(obs)

        ## detect phase changes and failed grasp
        if self.phase <= 41:
            self.door_moved = None
            self.segmented_pc = None
        self.phase = self.update_phase(self.phase, self.grasps[0], current_q,
                                  env_name=self.env_name, obs=obs)
        if self.phase not in self.phases:
            self.phases[self.phase] = 0
        self.phases[self.phase] += 1

        if self.phase == -1:
            # if self.maybe_open_left:
            #     self.is_open_right = not self.is_open_right
            #     self.maybe_open_left = False
            self.grasps.pop(0)
            if len(self.grasps) == 0:
                self.grasps = self.get_grasps(obs, num=20)
            next_pre, next_grasp, next_post = self.grasps[0]
            next_grasp = list(next_grasp)
            next_grasp[0] += 0.05
            self.grasps[0] = (next_pre, tuple(next_grasp), next_post)
            self.phase = 0
        # elif self.phase == 31:
        #     self.grasps[0] = get_push(self.start_obs)

        target_q, target_gripper = self.get_target_qpos(self.phase, self.grasps[0],
                                                   env_name=self.env_name, obs=obs)

        if self.phase == 44:
            self.reset_position_controllers()

        
        target_base_vel = self.base_controller.control(current_q[:3], target_q[:3])
        target_arm_vel = self.arm_controller.control(current_q[3:], target_q[3:])
        target_vel = list(target_base_vel)+list(target_arm_vel)

        action = target_vel
        action[11:13] = target_gripper
        if self.phase == 39:
            action[:2] = target_q[:2]
        elif self.phase in [43, 44, 45]:
            # open_left = not check_open_right()
            move_right = current_q[1] > target_q[1]
            move_left = not move_right
            straight_speed = 20
            if self.phase == 43:  ## going back enough to avoid collision
                x,y = target_q[:2] - current_q[:2]
                far = abs(y) > 0.4 ## np.sqrt(x**2+y**2) > 0.4
                action[2:] = [0] * len(action[2:])
                action[0] = - straight_speed * (-1) ** move_right  ## open_left
                action[1] = 0
                # if not self.is_open_right and self.check_away_from_handle(obs, epsilon=0.5):
                #     action[1] = straight_speed/10
                if far:
                    action[1] = - straight_speed/10 * (-1) ** move_right
            if self.phase == 44:  ## aligning y to be the offset from door rim
                action[2:] = [0] * len(action[2:])
                action[0] = 0
                action[1] = straight_speed * 2
            if self.phase == 45:  ## going into the gap
                action[2:] = [0] * len(action[2:])
                action[0] = straight_speed * 2 * (-1) ** move_left  ## open_left
                # print(abs(self.previous_wedge_x-current_q[0]))
                if(abs(self.previous_wedge_x-current_q[0])<0.0003):
                    action[1] = -1
                else:
                    action[1] = 0

                self.previous_wedge_x = current_q[0]
            # print('     current x,y vel', action[:2])
            # print()
        # print('         gripper vel', target_gripper)/

        diff = np.linalg.norm(current_q[:11] - target_q[:11])
        print(f'step {self.step}  | phase {self.phase}, diff in q {diff}')

        self.step += 1
    
        self.q_last = copy.deepcopy(current_q)
        self.a_last = copy.deepcopy(action)
        return action