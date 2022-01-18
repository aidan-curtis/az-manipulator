import sys
import importlib
sys.path.append('/opt/drake/lib/python3.8/site-packages')

import os
from os.path import join
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(join(dir_path, '..', 'drake_examples'))
# manipulation = importlib.import_module("drake_examples.manipulation")

import numpy as np

from controller import drake_ik

level = 3
num = 5
PREGRASP_DISTANCE = 0.8

VISUALIZE = False
n_key_points = 20

def fit_plane(xyzs):
    '''
    Args:
      xyzs is (N, 3) numpy array
    Returns:
      (4,) numpy array
    '''
    if isinstance(xyzs, list):
        xyzs = np.asarray(xyzs)
    #         print('min', np.min(xyzs, axis=0)[1])
    #         print('max', np.max(xyzs, axis=0)[1])
    #         print('mean', np.mean(xyzs, axis=0)[1])
    #         print()
    center = np.mean(xyzs, axis=0)
    cxyzs = xyzs - center
    U, S, V = np.linalg.svd(cxyzs)
    normal = V[-1]              # last row of V
    d = -center.dot(normal)
    plane_equation = np.hstack([normal, d])
    return plane_equation

def get_ranges(pc):
    if isinstance(pc, list):
        pc = np.asarray(pc)
    return np.max(pc, axis=0) - np.min(pc, axis=0)

def clean_box(pc, front_face=None, ext=50):

    x_min, y_min, z_min = np.min(pc, axis=0)
    x_max, y_max, z_max = np.max(pc, axis=0)
    x_range, y_range, z_range = get_ranges(pc)
    pc_topless = [p for p in pc if abs(p[2 ] -z_max) > z_range /50]

    if front_face == None:
        front_face = [p for p in pc_topless if abs(p[0 ] -x_min) < x_range /20]
    right_face = [p for p in pc_topless if abs(p[1 ] -y_min) < y_range /20]
    left_face = [p for p in pc_topless if abs(p[1 ] -y_max) < y_range /20]

    a, b, c, d = fit_plane(front_face)
    x_min = - d / a

    #     a, b, c, d = fit_plane(right_face)
    #     print('right_face', a, b, c, d)
    #     y_min = - d / b
    #     a, b, c, d = fit_plane(left_face)
    #     print('left_face', a, b, c, d)
    #     y_max = - d / b

    #     return [p for p in pc if p[1] > y_min * 1.02]
    #     return right_face + left_face

    y_min = np.mean(right_face, axis=0)[1]
    y_max = np.mean(left_face, axis=0)[1]
    body_pc = [p for p in pc if p[0] >= x_min]
    return body_pc, x_min, y_min, y_max

def clean_door(pc):
    a, b, c, d = fit_plane(pc)
    x_min = - d / a
    y_min = np.min(pc, axis=0)[1]
    y_max = np.max(pc, axis=0)[1]
    return pc, x_min, y_min, y_max

def visualize_arr(arr, name='', color=None):
    if not VISUALIZE: return
    import open3d as o3d
    if color == None:
        color = Rgba(1, 0, 0)
    file_name = '../temp/visualize_pc.txt'
    with open(file_name, 'w') as f:
        f.write('\n'.join([' '.join([str(m) for m in n]) for n in arr]))
    pcd = o3d.io.read_point_cloud(file_name, format='xyz')
    points = np.asarray(pcd.points).T
    cloud = PointCloud(points.shape[1])
    cloud.mutable_xyzs()[:] = points
    meshcat.SetObject(name, cloud, point_size=0.05, rgba=color)

def get_pc(obs):

    pc = []
    pc_color = []
    grasp_pc = []
    door_pc = []
    body_pc = []
    robot_pc = []

    for i in range(obs['pointcloud']['seg'].shape[0]):
        # Handle segmentation
        if (obs['pointcloud']['seg'][i, 0]):
            grasp_pc.append(obs['pointcloud']['xyz'][i])
            pc.append(obs['pointcloud']['xyz'][i])
            pc_color.append([1, 0, 0])

        # Door segmentation
        if (obs['pointcloud']['seg'][i, 1] and not obs['pointcloud']['seg'][i, 0]):
            door_pc.append(obs['pointcloud']['xyz'][i])
            pc.append(obs['pointcloud']['xyz'][i])
            pc_color.append([0, 1, 0])

        # Filter Floor
        if (obs['pointcloud']['xyz'][i, 2] > 0.1):

            # filter out the robot
            if not obs['pointcloud']['seg'][i, 2]:
                body_pc.append(obs['pointcloud']['xyz'][i])

                if not obs['pointcloud']['seg'][i, 0] and not obs['pointcloud']['seg'][i, 1]:
                    pc.append(obs['pointcloud']['xyz'][i])
                    pc_color.append([0, 0, 1])
            else:
                robot_pc.append(obs['pointcloud']['xyz'][i])

    segmented_pc = [grasp_pc, door_pc, body_pc, robot_pc]
    grasp_pc_arr = np.array(grasp_pc)
    door_pc_arr = np.array(door_pc)
    body_pc_arr = np.array(body_pc)
    robot_pc_arr = np.array(robot_pc)
    #     segmented_pc = [grasp_pc_arr, door_pc_arr, body_pc_arr, robot_pc_arr]

    # # TODO: Move these to a better place
    # if self.is_open_right == None or self.est_x_axis == None:
    #     self.is_open_right = self.check_open_right(segmented_pc)
    #     self.est_x_axis, self.est_y_axis = self.estimate_door_axis(segmented_pc, self.is_open_right)
    #
    # for xyz in door_pc:
    #     if abs(xyz[0] - self.est_x_axis) <= 0.02 and abs(xyz[1] - self.est_y_axis) <= 0.02:
    #         pc.append(xyz)
    #         pc_color.append([0, 0, 0])

    pc = np.array(pc)
    pc_color = np.array(pc_color)
    return pc, segmented_pc, pc_color, None

class Cabinet():

    def __init__(self, obs):

        pc, segmented_pc, pc_color, _ = get_pc(obs)

        self.grasp_pc, self.door_pc, self.body_pc, robot_pc = segmented_pc
        self.segmented_pc = segmented_pc
        self.handle_horizontal = None
        self.obs_count = 0

    #         visualize_arr(self.grasp_pc, 'handle before', color=Rgba(1, 0, 0))
    #         visualize_arr(hinge_pc, 'hinge before', color=Rgba(0, 0, 1))
    #         visualize_arr(self.body_pc, 'body before', color=Rgba(0, 0, 1))


    def estimate_params(self):
        ## clean the pc of the cabinet so we get better estimates for the hinge
        self.handle_horizontal = self.check_handle_horizontal(self.grasp_pc)
        self.is_open_right = self.check_open_right(self.segmented_pc)

        #         self.body_pc, self.x_min, self.y_min, self.y_max = clean_box(self.body_pc) ## , self.door_pc
        self.boor_pc, self.x_min, self.y_min, self.y_max = clean_door(self.door_pc)
        self.p_Whandle = self.estimate_handle()
        self.p_Whinge = self.estimate_hinge()
        hinge_pc = [p for p in self.door_pc if np.linalg.norm(self.p_Whinge[:2] - p[:2] ) <0.01]

        if VISUALIZE:
            visualize_arr(self.body_pc, 'body cleaned', color=Rgba(0, 1, 0))
            visualize_arr(self.door_pc, 'door merged', color=Rgba(1, 1, 0))
            visualize_arr(self.grasp_pc, 'handle merged', color=Rgba(1, 0, 0))
            visualize_arr(hinge_pc, 'hinge cleaned', color=Rgba(0, 0, 1))
        self.w, self.l, self.h = self.estimate_dimensions()

    def estimate_hinge(self):
        if self.is_open_right:  ## find the max of y direction
            y_axis = self.y_min
        else:
            y_axis = self.y_max
        x_axis = self.x_min
        #         z = np.mean(self.door_pc, axis=0)[2]

        return (x_axis, y_axis, self.p_Whandle[-1])  ## same level as the handle

    def estimate_handle(self):
        """ only the front half of the points"""
        grasp_arr = np.asarray(self.grasp_pc)
        x_mean, y_mean, z_mean = np.mean(grasp_arr, axis=0)

        x_min = np.min(grasp_arr, axis=0)[0]
        x_range, y_range, z_range = get_ranges(self.grasp_pc)
        handle_tip = np.asarray([p for p in self.grasp_pc if abs(p[0] - x_min ) < 0.5 * x_range])

        if VISUALIZE:
            visualize_arr(handle_tip, 'handle tip', color=Rgba(0, 0, 0))

        x_med, y_med, z_med = np.min(handle_tip, axis=0) + get_ranges(handle_tip) / 2
        p_handle = x_med, y_med, z_mean
        print('p_handle', p_handle)
        return p_handle

    def estimate_dimensions(self):
        ## add RANSAC
        x_range, y_range, z_range = get_ranges(self.body_pc)

        self.w = x_range / 2
        self.l = y_range / 2
        self.h = z_range / 2
        #         print(self.w, self.l, self.h)

        return self.w, self.l, self.h

    def add_obs(self, obs):

        pc, segmented_pc, pc_color, _ = get_pc(obs)
        grasp_pc, door_pc, body_pc, robot_pc = segmented_pc

        self.grasp_pc = self.grasp_pc + grasp_pc
        # self.door_pc = self.door_pc + door_pc
        # self.body_pc = self.body_pc + body_pc
        self.segmented_pc = self.grasp_pc, self.door_pc, self.body_pc, robot_pc
        self.obs_count += 1

    def check_handle_horizontal(self, grasp_pc):
        """ check the ratio of y_range / x_range """
        #         if self.handle_horizontal == None:
        x_range, y_range, z_range = get_ranges(grasp_pc)
        if y_range > 2* x_range and y_range > 2 * z_range:
            handle_horizontal = True
        elif z_range > 2 * x_range and z_range > 2 * y_range:
            handle_horizontal = False
        else:
            handle_horizontal = False
        #         self.handle_horizontal = handle_horizontal
        print('check_handle_horizontal', handle_horizontal)
        return handle_horizontal

    def check_open_right(self, segmented_pc):
        """ return True if the door opens to the right side """
        handle_pc, door_pc, body_pc, robot_pc = segmented_pc
        door_med = (np.min(door_pc, axis=0)[1] + np.max(door_pc, axis=0)[1]) / 2
        body_med = (np.min(body_pc, axis=0)[1] + np.max(body_pc, axis=0)[1]) / 2
        handle_med = (np.min(handle_pc, axis=0)[1] + np.max(handle_pc, axis=0)[1]) / 2

        ## if the handle is horizontal, the door body
        ## should lie on the right half of the object body
        if self.handle_horizontal:
            if door_med < body_med:
                result = True
            else:
                result = False

        ## if the handle is vertical, the center of the handle on the y axis
        ## should be to the left of the center of the door on the y axis
        else:
            if handle_med > door_med:
                result = True
            else:
                result = False
        print('check_open_right', result)
        return result

    def get_post_lst(self, last_q=[], PREGRASP_DISTANCE=0.4, verbose=False, VIZ_ALL_POSES=True):

        from pydrake.all import (
            RigidTransform, RotationMatrix, RollPitchYaw, PointCloud, Rgba, Cylinder
        )
        from manipulation import running_as_notebook, FindResource
        from manipulation.meshcat_cpp_utils import StartMeshcat, AddMeshcatTriad

        from pydrake.all import (
            DiagramBuilder, MeshcatVisualizerCpp, MeshcatVisualizerParams,
            Simulator, FindResourceOrThrow,
            Parser, MultibodyPlant, RigidTransform, RollPitchYaw,
            PiecewisePolynomial, PiecewiseQuaternionSlerp, RotationMatrix, Solve,
            TrajectorySource, ConstantVectorSource
        )
        from pydrake.multibody import inverse_kinematics
        from pydrake.trajectories import PiecewisePolynomial
        from manipulation.meshcat_cpp_utils import (
            StartMeshcat, AddMeshcatTriad
        )

        if VISUALIZE:
            meshcat.Delete()
        self.estimate_params()
        LEFT = not self.is_open_right

        X_Whinge = RigidTransform(self.p_Whinge)
        X_Whandle = RigidTransform(RollPitchYaw(0, 0, np.pi).ToRotationMatrix(), self.p_Whandle)

        r = RollPitchYaw(np.pi, np.pi / 2, 0).ToRotationMatrix()
        initial_pose = X_Whandle.multiply(RigidTransform(r, [PREGRASP_DISTANCE, 0, 0]))

        #     AddMeshcatTriad(meshcat, path='hinge', X_PT=X_Whinge) ## , opacity=.2
        #     AddMeshcatTriad(meshcat, path='handle', X_PT=X_Whandle) ## , opacity=.2
        #     AddMeshcatTriad(meshcat, path='pregrasp', X_PT=initial_pose) ## , opacity=.2

        p_Whinge = np.asarray(self.p_Whinge)
        p_Whandle = np.asarray(self.p_Whandle)

        p_Whinge_handle = p_Whandle - p_Whinge
        r_Rhinge_handle = np.linalg.norm(p_Whandle - p_Whinge)  # distance between handle and hinge.

        theta_Rhinge_handle = np.arctan2(p_Whinge_handle[1], p_Whinge_handle[0])
        angle_end = np.pi * 6/5
        if LEFT:
            angle_end = -np.pi * 6/5

        # Interpolate pose for opening doors.
        def InterpolatePoseOpen(t):
            # Start by interpolating the yaw angle of the hinge.
            angle_start = theta_Rhinge_handle
            theta = angle_start + (angle_end - angle_start) * t

            # Convert to position and rotation.
            p_Whandle = r_Rhinge_handle * np.array([np.cos(theta), np.sin(theta), 0]) + p_Whinge

            # Add some offset here to account for gripper yaw angle.
            R_Whandle = RollPitchYaw(0, 0, theta).ToRotationMatrix()
            if LEFT:
                R_Whandle = RollPitchYaw(0, 0, theta - np.pi).ToRotationMatrix()
            X_Whandle = RigidTransform(R_Whandle, p_Whandle)

            #     AddMeshcatTriad(meshcat, path='X_Whandle', X_PT=X_Whandle, opacity=.2)

            # Add a little offset to account for gripper.
            p_handleG = np.array([0., 0.1, 0])  ## np.array([0., 0.1, 0.])
            R_handleG = RollPitchYaw(0, np.pi / 2, -np.pi / 2).ToRotationMatrix()
            X_handleG = RigidTransform(R_handleG, p_handleG)
            X_WG = X_Whandle.multiply(X_handleG)
            if self.handle_horizontal:
                X_WG = X_WG.multiply(RigidTransform(RollPitchYaw(0, 0, -np.pi / 2).ToRotationMatrix()))

            return X_WG

        # Interpolate pose for opening doors.
        def HandlePose(t):
            # Start by interpolating the yaw angle of the hinge.
            angle_start = theta_Rhinge_handle
            theta = angle_start + (angle_end - angle_start) * t
            # Convert to position and rotation.
            p_Whandle = r_Rhinge_handle * np.array([np.cos(theta), np.sin(theta), 0]) + p_Whinge
            # Add some offset here to account for gripper yaw angle.
            R_Whandle = RollPitchYaw(0, 0, theta).ToRotationMatrix()
            X_Whandle = RigidTransform(R_Whandle, p_Whandle)
            return X_Whandle

        # AddMeshcatTriad(meshcat, path=str(1), X_PT=HandlePose(0), opacity=.2)
        if VISUALIZE and not VIZ_ALL_POSES:
            AddMeshcatTriad(meshcat, path=str(0), X_PT=InterpolatePoseOpen(0), opacity=.2)
        #     AddMeshcatTriad(meshcat, path=str(0.3), X_PT=InterpolatePoseOpen(0.3), opacity=.2)

        ## now phase 102 just dashes straight into the grasp position
        initial_pose = RigidTransform([-PREGRASP_DISTANCE, 0, 0]).multiply(InterpolatePoseOpen(0))

        ## Interpolate Pose for entry.
        def make_gripper_orientation_trajectory():
            traj = PiecewiseQuaternionSlerp()
            traj.Append(0.0, initial_pose.rotation())
            traj.Append(5.0, InterpolatePoseOpen(0.0).rotation())
            return traj

        def make_gripper_position_trajectory():
            traj = PiecewisePolynomial.FirstOrderHold(
                [0.0, 5.0],
                np.vstack([[initial_pose.translation()],
                           [InterpolatePoseOpen(0.0).translation()]]).T)
            return traj

        def InterpolatePoseEntry(t):
            return RigidTransform(RotationMatrix(entry_traj_rotation.value(t)),
                                  entry_traj_translation.value(t))

        ## interpolate rotation and position separately
        entry_traj_rotation = make_gripper_orientation_trajectory()
        entry_traj_translation = make_gripper_position_trajectory()

        # Wrapper function for end-effector pose. Total time: 11 seconds.
        def InterpolatePose(t):
            if (t < 5.0):
                # Duration of entry motion is set to 5 seconds.
                return InterpolatePoseEntry(t)
            elif (t >= 5.0) and (t < 6.0):
                # Wait for a second to grip the handle.
                return InterpolatePoseEntry(5.0)
            else:
                # Duration of the open motion is set to 5 seconds.
                return InterpolatePoseOpen((t - 6.0) / 5.0)

        # Visualize our end-effector nominal trajectory.
        t_lst = np.linspace(0, 11, n_key_points)  ## 5, 10)  ##
        pose_lst = []
        # last_q = None
        for t in t_lst:
            if t > 0 and t < 5: continue
            if VISUALIZE and VIZ_ALL_POSES:
                AddMeshcatTriad(meshcat, path=str(t), X_PT=InterpolatePose(t), opacity=.1)

            pose = InterpolatePose(t)
            if len(last_q) == 0:
                last_q = [ -1.37, 0.58, 0.0, 0.0, 0.1, 1.36, 0.0, -0.28, -0.15, 3.0, 0.29, 0.0, 0.0 ]
            q = drake_ik(pose.translation(), pose.rotation(), list(last_q), verbose=False,
                         hand_link="right_panda_hand")
            pose_lst.append(q)
            last_q = q

        #     # Create gripper trajectory.
        #     gripper_t_lst = np.array([0., 5., 6., 11.])
        #     gripper_knots = np.array([0.05, 0.05, 0., 0.]).reshape(1, 4)
        #     g_traj = PiecewisePolynomial.FirstOrderHold(gripper_t_lst, gripper_knots)
        return pose_lst


# pose_lst = level_to_traj(level, num, PREGRASP_DISTANCE, VIZ_ALL_POSES=True)

import gym
import pickle
from mani_skill import env
import time
import os


def show_level(level_idx, VIZ_ALL_POSES=True):
    env = gym.make('OpenCabinetDoor-v0')
    env.set_env_mode(obs_mode='pointcloud', reward_type='sparse')
    for level_idx in range(level_idx, 1000):
        obs = env.reset(level=level_idx)
        print('\n#### Level {:d}'.format(level_idx))
        file_names = []

        cabinet = Cabinet(obs)

        for i_step in range(100):
            env.render('human')  # a display is required to use this function, rendering will slower the running speed
            # action = env.action_space.sample()
            action = [0 for _ in range(13)]
            action[0] = 1
            action[1] = -1
            obs, reward, done, info = env.step(action)  # take a random action
            cabinet.add_obs(obs)

            file_name = f'../temp/{level_idx}_{i_step}.pkl'
            file_names.append(file_name)
            with open(file_name, 'wb') as handle:
                pickle.dump(obs, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if i_step >= num:
                break

        #         pose_lst = level_to_traj(level_idx, num, PREGRASP_DISTANCE, VIZ_ALL_POSES)
        pose_lst = cabinet.get_post_lst(VIZ_ALL_POSES=VIZ_ALL_POSES)
        print(pose_lst)
        for file in file_names:
            os.remove(file)
        env.close()
        time.sleep(2)

        break


# show_level(1, VIZ_ALL_POSES=False)