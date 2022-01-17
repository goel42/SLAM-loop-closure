import numba
import pandas as pd
import os
import numpy as np
from scipy.spatial.transform import Rotation as R


from ati.slam.yelli import utils, grid

GRID_START_SIZE = (128, 128, 128)
GRID_RESIZE = 1.5
GRID_RES = 0.02

GRID_START_VALUE = 0.0

class Grid3D:
    def __init__(self, grid_alpha=1, grid_res = GRID_RES):
        self.grid = np.full(GRID_START_SIZE, GRID_START_VALUE, dtype=np.uint8)
        self.origin = np.zeros(3)
        self.grid_alpha = grid_alpha
        self.grid_res = grid_res
        

    def set_grid(self, grid):
        self.grid = grid

    def _resize_map(self, origin_shift, new_size):
        old_grid, self.grid = self.grid, np.full(new_size, GRID_START_VALUE, dtype=self.grid.dtype)
        (sx, sy, sz), (ex, ey, ez) = origin_shift, old_grid.shape
        self.grid[sx : sx + ex, sy : sy + ey, sz : sz + ez] = old_grid
        self.origin += origin_shift

    def _maybe_resize_map(self, frame):
        """frame here has shape Nx3 i.e. N 3D point cordinates"""
        lower, upper = np.min(frame, axis=0), np.max(frame, axis=0)

        if np.any(lower < 0) or np.any(upper >= self.grid.shape):
            origin_shift = -np.minimum(lower, 0)
            origin_shift = origin_shift + (origin_shift != 0) * (
                np.array(self.grid.shape) * (GRID_RESIZE - 1)
            )
            new_size = np.maximum(self.grid.shape, upper * GRID_RESIZE) + origin_shift

            origin_shift, new_size = (
                origin_shift.astype(np.int32),
                new_size.astype(np.int32),
            )

            self._resize_map(origin_shift, new_size)
            return True
        return False
    
    def transform_world_to_grid(self, frame):
        return frame / self.grid_res + self.origin

    def transform_local_to_grid(self, frame, pose):
        """frame here has shape Nx3 i.e. N 3D point cordinates"""
        frame = transform_local_to_world(pose, frame) #TODO: Verify for type conversion this returns np.float (0 is 2e-16 etc so does that cause any problem?)
        frame = self.transform_world_to_grid(frame).astype(np.int32)
        return frame
    
    def insert_points(self, pose, frame):
        """Insert 2D frame into map"""
        if frame.shape[0] == 0:
            return
        frame_local = self.transform_local_to_grid(frame, pose)
        if self._maybe_resize_map(frame_local):
            print("blahblahcar")
            frame_local = self.transform_local_to_grid(frame, pose)
        x, y, z = frame_local.T
#         print("xyz", np.max(x),np.max(y),np.max(z))
#         print("grid shape", self.grid.shape)
#         print("pose", pose)
        self.grid[x, y, z] = np.minimum(self.grid[x, y, z] + self.grid_alpha, 255) #TODO: verify z lie on the same plane

    def search(self, frame, search_space, count_once=False):
        """Score of poses in search_space. Frame must be in local coordinates"""
        if not count_once:
            return search_fastv2(self.grid, self.origin, search_space, frame, self.grid_res)
        else:
            return search_fast_count_once(self.grid, self.origin, search_space, frame, self.grid_res)


def rotate(points, theta):
    theta -= np.pi / 2 #WHY?
    r = R.from_quat([0, 0, np.sin(theta/2), np.cos(theta/2)])
    points_rotated=r.apply(points)
    return points_rotated

# TODO: Fix rotation matrix and move ?~@/2 out here
def transform_local_to_world(pose, points):
    """Transform local coordinates to world coordinates"""
    x, y, z, theta = pose
    points_rotated = rotate(points,theta) #TODO: verify
    return points_rotated + np.array([x, y, z])

# TODO: Generate grid and rotate it later - don't regenerate it all the time
def grid_space(center, x, y, theta, num_x=9, num_y=15, num_t=9):
    """center is the global pose of the grid. x, y, theta are 2-tuples of (min, max)"""
    search_space = np.array(
        np.meshgrid(
            np.linspace(*x, num=num_x),
            np.linspace(*y, num=num_y),
            0,
            np.linspace(*theta, num=num_t),
        )
    ).T.reshape(-1, 4)
    search_space[:, :3] = transform_local_to_world(center, search_space[:, :3])
    search_space[:, 3] += center[3]

    return search_space
    
@numba.njit(fastmath=True, parallel=True)
def search_fastv2(grid, grid_origin, poses, frame, grid_res):
    scores = np.zeros(poses.shape[0])
    for i in numba.prange(poses.shape[0]):
        scores[i] = score_function(grid, grid_origin, poses[i, :], frame, grid_res)
    return scores

@numba.njit(fastmath=True)
def score_function(grid, grid_origin, pose, frame, grid_res):
    """Score of frame in map. Frame must be in local coordinates."""
    score = 0

    px, py, pz, ptheta = pose[0], pose[1], pose[2], pose[3] - np.pi / 2
    c, s = np.cos(ptheta), np.sin(ptheta)

    for i in range(frame.shape[0]):
        x, y, z = frame[i, 0], frame[i, 1], frame[i, 2]
        ix = int(np.ceil(((c * x - s * y) + px) / grid_res + grid_origin[0]))
        iy = int(np.ceil(((s * x + c * y) + py) / grid_res + grid_origin[1]))
        iz = int(np.ceil((z + pz)/ grid_res + grid_origin[2])) #TODO: verify
        if (ix >= 0) and (ix < grid.shape[0]) and (iy >= 0) and (iy < grid.shape[1]) and (iz >= 0) and (iz < grid.shape[2]):#TODO: verify
            score += grid[ix, iy, iz]


    return score
    

    
    
    
    
    
    
