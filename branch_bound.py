import pandas as pd
import os
import numpy as np
import numba
from grid2d import Grid2D
from ati.slam.yelli import utils, grid

#@numba.njit(fastmath=True, parallel=True)
def compute_hierarchical_grid(grid2d, num_levels):
    hgrids=[]
    levels = np.arange(1, num_levels +1)
    grid_shape = grid2d.grid.shape
    prev_grid = grid2d
    for level in levels:
        grid_shape = prev_grid.grid.shape
        #win = 2**level
        shape = (int(np.ceil(grid_shape[0]/2)), int(np.ceil(grid_shape[1]/2)))
        hgrid_res = 2 * prev_grid.grid_res
        grid = Grid2D(grid_res = hgrid_res)
        grid.set_grid(np.full(shape, 0, dtype=np.uint8))
        grid.origin = (prev_grid.origin/2).astype(np.int32)
        #grid_arr = np.full(shape, 0, dtype=np.uint8)
        grid.set_grid(compute_half_res_grid(grid.grid, prev_grid.grid))
        hgrids.append(grid)
        prev_grid = grid
    return hgrids
                    
@numba.njit(fastmath=True, parallel=True)                    
def compute_half_res_grid(half_res_grid, full_res_grid):
    for i in range(half_res_grid.shape[0]):
        for j in range(half_res_grid.shape[1]):
            xrange = 2*i, min(2*i+2, full_res_grid.shape[0])
            yrange = 2*j, min(2*j+2, full_res_grid.shape[1])
            #half_res_grid[i,j] = np.max(full_res_grid[xrange[0]: xrange[1], yrange[0]: yrange[1]])
            #Below is faster
            half_res_grid[i,j] = max(full_res_grid[xrange[0]: xrange[1], yrange[0]: yrange[1]].flatten())
            #if i > 20:
            #    break
    return half_res_grid


def hierarchical_search(hgrids,pose_estimate, frame, x_search_window,
                        y_search_window, angle_search_window, count_once, max_range=60, score_th = 0.2, debug=False):
    best_pose = pose_estimate
    for i in reversed(range(len(hgrids))):
        grid = hgrids[i]
        angular_res = grid.grid_res/max_range
        num_x = int(np.ceil((x_search_window[1] - x_search_window[0])/grid.grid_res))
        num_y = int(np.ceil((y_search_window[1] - y_search_window[0])/grid.grid_res))
        num_t = int(np.ceil(angle_search_window[1] - angle_search_window[0]/angular_res))
        
        search_space = utils.grid_space(best_pose, 
                                        x_search_window, y_search_window, angle_search_window, num_x, num_y, num_t)
        if count_once:
            scores, num_visited = grid.search(frame, search_space, count_once=True)
            best = np.argmax(scores)
            best_normalized_score = scores[best]/num_visited[best]
        else:
            scores = grid.search(frame, search_space, count_once=False)
            best = np.argmax(scores)
            best_normalized_score = scores[best]/len(frame)
        #print(f"Score for {i}th grid is {best_normalized_score}")
        if best_normalized_score < score_th:
            if debug:
                print(f"Score below thresold {score_th} score :{best_normalized_score}")
            return None, best_normalized_score
        best_pose = search_space[best]
        if debug:
            print(f"For {i}th grid num_x:{num_x}, num_y:{num_y}, num_t:{num_t} , grid_res {grid.grid_res}, anguler res {angular_res}")
            print(f"angle_search_window {angle_search_window}")
            print(f"Score for {i}th grid is {best_normalized_score}")
            print(f"Best_pose for {i}th grid {best_pose}")
        #x_search_window = (best_pose[0] - grid.grid_res, best_pose[0])
        #y_search_window = (best_pose[1] - grid.grid_res, best_pose[1])
        x_search_window = (-grid.grid_res, grid.grid_res)
        y_search_window = (-grid.grid_res, grid.grid_res)
        angle_search_window = (-angular_res,angular_res)
    return best_pose,best_normalized_score 
#%time gs = compute_hierarchical_grid(smap.grid, num_levels=8)