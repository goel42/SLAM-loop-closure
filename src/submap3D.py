import sys
sys.path.append("/home/anujraaj/robo/SLAM-loop-closure/src/")

from grid3D import Grid3D
import numpy as np
import pandas as pd
import os
import pickle 
import zstandard

class Submap:
    def __init__(self, pose, grid_res, grid_alpha):
        self.pose = pose.copy()
        self.grid_alpha = grid_alpha
        self.grid_res = grid_res
        self.grid = Grid3D(self.grid_alpha, self.grid_res )
        self.num_insertions = 0
        self.finished = False
        self.node_ids = []
        self.frame_ids = []
        self.local_insertion_poses = None
        self.iscompressed = False
        
    def search(self, frame, search_space, count_once = True):
        return self.grid.search(frame, search_space, count_once=count_once)
    
    def insert_points(self, frame, pose, frame_id, node_id):        
        if not self.finished:
            self.num_insertions +=1
            self.grid.insert_points(pose, frame)
            self.node_ids.append(node_id)
            self.frame_ids.append(frame_id)
            if self.local_insertion_poses is None:
                self.local_insertion_poses = pose.reshape(1,4)
            else:
                self.local_insertion_poses = np.concatenate((self.local_insertion_poses, pose.reshape(1,4)), axis = 0)
        else:
            print("Submap update is finished. Not inserting ")
            
    def finish(self):
        self.finished = True
        self.compress_data()
    
    def compress_data(self):
        cctx = zstandard.ZstdCompressor()
        self.grid = pickle.dumps(self.grid)
        self.grid = cctx.compress(self.grid)
        self.iscompressed = True
        
    def decompress_data(self):
        dctx = zstandard.ZstdDecompressor()
        self.grid = dctx.decompress(self.grid)
        self.grid = pickle.loads(self.grid)
        self.iscompressed = False