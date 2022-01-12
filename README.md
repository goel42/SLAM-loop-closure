# SLAM-loop-closure
experiments with finding loop closure constraints using Hierarchical search from Cartographer and optimization using G2O
![image](https://user-images.githubusercontent.com/12610586/149178703-8f79e67a-2fee-47ad-8791-49e11f093ae8.png)
![image](https://user-images.githubusercontent.com/12610586/149178765-54483c0c-e9a7-4397-a4bd-e1ac61e18668.png)


## Submaps:

Submaps consist of a fixed number of inserted lidar frames. There are two submaps in operation at any point except for the initialization. 

1. $n :$ Number of frames to be inserted into the submap.
2. submap_pose: Pose of the submap in the global frame.
3. node_ids and frame_ids: index of global poses inserted into the submaps and the frame_ids corresponding to the frames inserted.
4. Initialization: There is only one submap. We localize and insert frames up to until $n/2$ frames are inserted.
5. Once n/2 frames are inserted into the 1st submap we create the 2nd submap. From here onwards there will always be two submaps active.
6. We keep localizing in 1st submap and insert frames into both 1st submap and 2nd submap (can be thought of as initialization of the 2nd submap).
7. Once $n$  frames are inserted into the 1st submap we consider the submap to be finished and is taken out of consideration and added to the list of finished submaps.
8. At this point, the 2nd submap has $n/2$ frames inserted and we create the 3rd submap.
9. Frames are now localized in the 2nd submap and inserted into both 2nd and 3rd submaps (can be thought of as initialization of the 3rd submap).
10. Then the same process continues for the subsequent submaps, once submaps have $n$ frames inserted then they are added to the list of finished submaps.

## Loop Closing Constraints

We want to find the constraints between nodes so we can add them to the graph optimization problem later. To find the constraints we:

1. Search node frames within a range of the submaps and threshold by the score. We search over large search spaces order of few meters and angular search space of tens of degrees.
2. Given that the score meets our threshold we add the constraints between the submap_node_id and the node_id (corresponding to the frame that is used for the search).

# Hierarchical Search

To search over large spaces efficiently we use a hierarchical grid to do the search.

### Computing hierarchical grid

1. Num levels: Number of levels of the grid to compute.
2. Compute each level map by doubling the resolution for each level.
3. For each grid point next resolution level, take the max value from the corresponding lower resolution level.

### Localizing in a hierarchical grid

1. Start from the grid with the largest resolution. The x-y grid spacing is corresponding to the larger grid. 
2. For a max range of d, the angular resolution is computed according to $d/grid\_res$.
3. For the next level, the x-y grid spacing corresponds to the grid in the previous larger resolution and the angular resolution of the previous grid.
4. Keep the scores in a sorted list. 
5. Pop the the search space and grid with the largest score and continue the search.
6. Continue until the grid with the smallest resolution is reached.

# Pose graph

**Constraints:** Given a graph topology, the constraints are the edges between the nodes of the graph (which themselves correspond to insertion poses ).

### Computing the constraints

We essentially compute constraints between the submaps and the nodes. The submap node corresponds to the 1st node that was inserted into the graph. 

1. For a particular node (which itself would generally already have connections to 2 submaps), we would like to check against all submaps and in general be able to distinguish between good and bad matches.
2. The scores were not very effective in distinguishing between good and bad matches if done with all the submaps.
3. Also computing the scores by localizing across all submaps for each inserted frame was too compute intensive.
4. So we only check for nodes that lie within a threshold distance from the submap node and use a strict score threshold to discard spurious constraints (in the process we also discard many good matches but tuning the score threshold has been unreliable). We also check if the pose is close to the middle of the submap with another distance threshold (It is expected that the middle would be the densest part of the submap).
5. There are still at least two ways to go about it. One is one-shot matching, if we expect the initial localization drift to be within our search threshold, we match the frames to submaps and add to posegraph problem and optimize over it.
Second is that if we expect the drift to be more than our search space, we go through the progression of nodes from the 1st node and optimize when we get a constraint (or fixed number of constraints), correct the graph and use the new poses to continue the process, this way we expect that as we go along the progression of nodes we are likely to stay within our search limits.

## Simple posegraph

Here the posegraph is a simple linkage of node poses. The red circles are the submap poses. The green lines are the generated loop closing constraints. Using g2o to optimize over such a graph generally leads to a map with thick edges as the optimal poses as per the scan matching algorithm is overridden. 

![Screenshot 2022-01-03 at 3.22.17 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c4837c4b-231c-4e86-bf9d-f7e4402fdfc5/Screenshot_2022-01-03_at_3.22.17_PM.png)

### Posegraph representing the submap structure

Here the posegraph follows the structure of the submap structure. The nodes are linked to the submaps (submap poses) they localize or are inserted into. Except for the initial nodes, all nodes are always linked to two submaps. Once the loop closing constraints are generated, we optimize over the graph using g2o, this results in better maps compared to the simple posegraph above. The relative constraints between the submap nodes and the ordinary nodes are better maintained in this setup. 

 

![Screenshot 2022-01-03 at 6.03.38 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/74f7ade7-d2a0-4d3e-b85a-a76eeb7776ea/Screenshot_2022-01-03_at_6.03.38_PM.png)

### Submap graph

Instead of having the graph of all inserted poses, we could have the submap graph where only the submap nodes exist, and once the loop closing constraints are found we convert it into submap to submap constraints. Thereafter the submaps could be combined directly. This hasn’t been pursued yet.

### G2O

g2opy is the package that was used for the pose graph optimization. It's a wrapper over the g2o library. The basic sample program was used with some minor modifications. One of the parameters to be passed is what is called an Information matrix for all of the constraints which can be used to denote our confidence in our estimates that is passed to the solver. This could be a function of the score (during the scan-submap matching) or some other metric. The score didn’t seem like a very reliable metric and the subsequently tweaking the information matrix (default was identity) wasn’t pursued thoroughly.

## Generating the map

Once we get the optimized poses we use those and the corresponding lidar frames to generate the map.



