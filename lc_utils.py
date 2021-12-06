import pandas as pd
import os
import numpy as np


def get_2D_rotation_mat(theta):
    cth = np.cos(theta)
    sth = np.sin(theta)
    return np.array([[cth, -sth],
                    [sth, cth]])

def get_angle_from_2D_rotation_mat(R):
    return np.arctan2(R[1,0], R[0,0])

def get_yelli_rotation(theta):
    th = theta - np.pi/2
    R = get_2D_rotation_mat(th)
    return R

def get_inverse_yelli_rotation(theta):
    return get_yelli_rotation(theta).T

def get_yelli_transform(pose):
    R = get_yelli_rotation(pose[2])
    t = pose[:2]
    return R,t

def get_inverse_yelli_transform(pose):
    R = get_inverse_yelli_rotation(pose[2])
    t = -R @ pose[:2].reshape(2,1)
    return R,t.flatten()

def normalize_pose(theta):
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    return theta

def get_inverse_yelli_pose(pose):
    #theta = -pose[2] + np.pi
    #theta = normalize_pose(theta)
    R,t = get_inverse_yelli_transform(pose) 
    theta = get_angle_from_2D_rotation_mat(R) + np.pi/2 
    theta = normalize_pose(theta)
    return np.array([t[0], t[1], theta])


#WHAT IS?
def combine_yelli_poses(p1, p2):
    R1,t1 = get_yelli_transform(p1)
    R2,t2 = get_yelli_transform(p2)
    R = R1 @ R2
    t = (t1.reshape(2,1) + R1 @ t2.reshape(2,1)).flatten()
    theta = get_angle_from_2D_rotation_mat(R) + np.pi/2
    theta = normalize_pose(theta)
    return np.array([t[0], t[1], theta])

def normalize_pose(theta):
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    return theta

def load_imu_data(dir_path):
    imu_data = pd.read_csv(os.path.join(dir_path, "imu.csv"))
    # imu_data.gyro_z = np.deg2rad(imu_data.gyro_z)
    return imu_data

from ati.perception.utils import quaternion_utils as qe
def imu_pose_estimate(imu_tracker, time, pose):
    dquat = imu_tracker.get_gyro_quaternion(time)
    dtheta = qe.get_roll_pitch_yaw_from_quaternion(dquat)[-1]
    pose_estimate = np.copy(pose)
    pose_estimate[2] += dtheta
    return pose_estimate