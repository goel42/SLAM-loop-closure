import numba
import pandas as pd
import os
import numpy as np

from ati.slam.yelli import utils, grid

GRID_START_SIZE = (128, 128)
GRID_RESIZE = 1.5
GRID_RES = 0.02

GRID_START_VALUE = 0.0

class Grid2D:
    def __init__(self, grid_alpha=1, grid_res = GRID_RES):
        self.grid = np.full(GRID_START_SIZE, GRID_START_VALUE, dtype=np.uint8)
        self.origin = np.zeros(2)
        self.grid_alpha = grid_alpha
        self.grid_res = grid_res
        

    def set_grid(self, grid):
        self.grid = grid

    def _resize_map(self, origin_shift, new_size):
        old_grid, self.grid = self.grid, np.full(new_size, GRID_START_VALUE, dtype=self.grid.dtype)
        (sx, sy), (ex, ey) = origin_shift, old_grid.shape
        self.grid[sx : sx + ex, sy : sy + ey] = old_grid
        self.origin += origin_shift

    def _maybe_resize_map(self, frame):
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
        frame_xy = frame[:, :-1]
        frame_xy = utils.transform_local_to_world(pose, frame_xy)
        frame_xy = self.transform_world_to_grid(frame_xy).astype(np.int32)
        return frame_xy
    
    def insert_points(self, pose, frame):
        """Insert 2D frame into map"""
        if frame.shape[0] == 0:
            return
        frame_xy = self.transform_local_to_grid(frame, pose)
        if self._maybe_resize_map(frame_xy):
            frame_xy = self.transform_local_to_grid(frame, pose)
        x, y = frame_xy.T
        self.grid[x, y] = np.minimum(self.grid[x, y] + self.grid_alpha, 255)

    def search(self, frame, search_space, count_once=False):
        """Score of poses in search_space. Frame must be in local coordinates"""
        if not count_once:
            return search_fastv2(self.grid, self.origin, search_space, frame, self.grid_res)
        else:
            return search_fast_count_once(self.grid, self.origin, search_space, frame, self.grid_res)

    def apply_median_filter(self, kernel_size):
        self.grid = scipy.ndimage.median_filter(self.grid, kernel_size)

    def linear_transform(self, t1=3, t2=20):
        self.grid[self.grid < t1] = 0
        self.grid[self.grid > t2] = t2
        
        
@numba.njit(fastmath=True, parallel=True)
def search_fastv2(grid, grid_origin, poses, frame, grid_res):
    scores = np.zeros(poses.shape[0])
    for i in numba.prange(poses.shape[0]):
        scores[i] = score_function(grid, grid_origin, poses[i, :], frame, grid_res)
    return scores

@numba.njit(fastmath=True, parallel=True)
def search_fast_count_once(grid, grid_origin, poses, frame, grid_res):
    scores = np.zeros(poses.shape[0])
    for i in numba.prange(poses.shape[0]):
        scores[i] = score_function_count_once(grid, grid_origin, poses[i, :], frame, grid_res)
    return scores


@numba.njit(fastmath=True)
def score_function_count_once(grid, grid_origin, pose, frame, grid_res):
    """Score of frame in map. Frame must be in local coordinates."""
    score = 0

    px, py, ptheta = pose[0], pose[1], pose[2] - np.pi / 2
    c, s = np.cos(ptheta), np.sin(ptheta)
    visited = set()
    for i in range(frame.shape[0]):
        x, y, z = frame[i, 0], frame[i, 1], frame[i, 2]
        ix = int(((c * x - s * y) + px) / grid_res + grid_origin[0])
        iy = int(((s * x + c * y) + py) / grid_res + grid_origin[1])
        if (ix,iy) in visited:
            continue
        if (ix >= 0) and (ix < grid.shape[0]) and (iy >= 0) and (iy < grid.shape[1]):
            score += grid[ix, iy]
        visited.add((ix,iy))

    return score

@numba.njit(fastmath=True)
def score_function(grid, grid_origin, pose, frame, grid_res):
    """Score of frame in map. Frame must be in local coordinates."""
    score = 0

    px, py, ptheta = pose[0], pose[1], pose[2] - np.pi / 2
    c, s = np.cos(ptheta), np.sin(ptheta)

    for i in range(frame.shape[0]):
        x, y, z = frame[i, 0], frame[i, 1], frame[i, 2]
        ix = int(np.ceil(((c * x - s * y) + px) / grid_res + grid_origin[0]))
        iy = int(np.ceil(((s * x + c * y) + py) / grid_res + grid_origin[1]))
        
        if (ix >= 0) and (ix < grid.shape[0]) and (iy >= 0) and (iy < grid.shape[1]):
            score += grid[ix, iy]


    return score



#%timeit sc=search_fast_count_once(active_submaps[0].grid.grid, active_submaps[0].grid.origin, search_space, frame_z, active_submaps[0].grid.grid_res )
#sc/len(frame_z)