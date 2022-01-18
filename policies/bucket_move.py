

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
import pybullet as p 
from pointcloud_utils import dist, dist3d
from controller import get_robot_qpos_from_obs, open_gripper, equal, \
    closed_gripper, open_gripper, dual_drake_ik

# Hugging Bucket
class BucketPolicy(DualAPolicy):

    def __init__(self, env_name):
        super().__init__()


    def reset(self):
        raise NotImplementedError


    def act(self, obs):
        raise NotImplementedError


    def is_cyan(self, color):
        cyan_max_thresh = 0.5
        cyan_min_thresh = 0.1
        return  color[0]<cyan_min_thresh and \
                color[1]>cyan_max_thresh and \
                color[2]>cyan_max_thresh
    def generate_grasps(self, grasp_pc):

        grasp_poses = []
        mean_point = np.mean(grasp_pc, axis=0)
        print(mean_point)
        for pt in grasp_pc:
            x_diff = pt[0]-mean_point[0]
            y_diff = pt[1]-mean_point[1]
        
            theta = math.atan(y_diff/x_diff)
            if(x_diff<0):
                theta+=math.pi

            r = R.from_quat((p.getQuaternionFromEuler([-theta+math.pi/2+math.pi, math.pi/2,0])))
            quat = list(r.as_quat())
            score = [0]
            pos = [pt[0], pt[1], pt[2]+0.02]
            grasp_poses.append(list(pos)+quat+score)

        grasp_poses.sort(key=lambda s: dist(s[:3], mean_point))

        # This is probably worse than just finding the grasp closest to the starting gripper position
        # grasp_pairs = []
        # # Find grasp pairs that are the furthest away from each other
        # for grasp1 in grasp_poses:
        #     max_grasp_dist = -float("inf")
        #     max_grasp_el = None
        #     for grasp2 in grasp_poses:
        #         pdiff = dist3d(grasp1[:3], grasp2[:3])
        #         if(pdiff>max_grasp_dist):
        #             max_grasp_dist = pdiff
        #             max_grasp_el = grasp2

        #     if(max_grasp_el is not None):
        #         grasp_pairs.append( (grasp1, max_grasp_el) )

        return grasp_poses

    def get_desired_theta(self, current_q):
        chairx, chairy = current_q[0], current_q[1]
        desired_theta = math.atan(chairy/chairx)
        if(chairx<0):
            desired_theta = desired_theta+math.pi

        desired_theta += math.pi
        desired_theta = desired_theta%(2*math.pi)
        return desired_theta


    def get_pc(self, obs):
        pc = []
        pc_color = []
        bucket_pc_pc_color = []
        min_bucket_height = 0.15
        for i in range(obs['pointcloud']['seg'].shape[0]):
            # Handle segmentation
            if(obs['pointcloud']['seg'][i, 0] == 0 and obs['pointcloud']['xyz'][i, 2] > min_bucket_height and not self.is_cyan(obs['pointcloud']['rgb'][i])):
                bucket_pc_pc_color.append((obs['pointcloud']['xyz'][i], obs['pointcloud']['rgb'][i]))
            # Filter Floor
            if(obs['pointcloud']['xyz'][i, 2] > 0.1):
                pc.append(obs['pointcloud']['xyz'][i])
                pc_color.append(obs['pointcloud']['rgb'][i])

        # TODO: Ransac on bucket points to get the rim of the bucket
        # z_thresh = 0.0005
        z_thresh = 0.0005
        ransac_points = 60
        # Step 1: Sort points by z value
        start_point = 0
        found_points = False
        while(not found_points):
            # Find highest rim point using ransac
            bucket_pc_pc_color = sorted(bucket_pc_pc_color, key=lambda d: d[0][2], reverse=True)
            for bpi, bucket_point in enumerate(bucket_pc_pc_color):
                if(bpi+ransac_points<len(bucket_pc_pc_color)):
                    base = min(bpi+ransac_points, len(bucket_pc_pc_color)-1) 
                    if(bucket_pc_pc_color[base][0][2]>bucket_point[0][2]-z_thresh):
                        start_point = bpi
                        break
            else:
                ransac_points-=5
                z_thresh+=0.0005
                continue

            print(start_point)
            # Extract rim points
            rim_pc = []
            rim_pc_color = []
            for rpi in range(start_point, len(bucket_pc_pc_color)-1):
                if(bucket_pc_pc_color[rpi][0][2]>bucket_pc_pc_color[start_point][0][2]-z_thresh):
                    rim_pc.append(bucket_pc_pc_color[rpi][0])
                    rim_pc_color.append(bucket_pc_pc_color[rpi][1])
                else:
                    break


            if(len(rim_pc)>0):
                found_points=True

            ransac_points-=5
            z_thresh+=0.0005


        pc = np.array(pc)
        pc_color = np.array(pc_color)
        rim_pc = np.array(rim_pc)
        rim_pc_color = np.array(rim_pc_color)

        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(rim_pc)
        # o3d.visualization.draw_geometries([pcd])

        # print("Num rim points: "+str(len(rim_pc)))

        return pc, rim_pc, pc_color, rim_pc_color
