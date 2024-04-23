from ast import arg
from math import degrees
import numpy as np
import os
import logging
import torch
import datetime
import cv2
import gym
import matplotlib as mpl
import sys
import math
import networkx as nx
import copy

from scipy.ndimage import rotate
from matplotlib import pyplot as plt
from PIL import Image
from networkx.classes.function import path_weight

from gibson2.utils.utils import parse_config, dist
from SAC.SAC import SAC
from occupancy_grid.occupancy_grid import visualize_path, ray_casting, inflate_grid
from frontier.frontier import get_frontiers, show_frontiers, frontier_path_cost
from gibson2.utils.constants import *
from encoder.obs_encoder import ObsEncoder
#from simple_env import Simple_env
from simple_env_original import Simple_env_original


logging.getLogger().setLevel(logging.WARNING)

#gc.set_threshold(0)

class Challenge:
    def __init__(self):
        self.config_file = CONFIG_FILE
        self.split = SPLIT
        self.episode_dir = EPISODE_DIR
        self.eval_episodes_per_scene = 100
        self.training_scenes = ["Beechwood_1_int", "Ihlen_0_int", "Ihlen_1_int", "Merom_0_int"]
        self.test_scenes = ["Benevolence_0_int", "Rs_int", "Wainscott_1_int"]
        self.valdiation_scene = ["Pomaria_0_int"]

    def plan(self, env, ped_inflated_grid):
        ped_cells = np.logical_and(env.inflated_grid==1, ped_inflated_grid==0)
        ped_cells_index = np.vstack((np.where(ped_cells==1)[0], np.where(ped_cells==1)[1])).T 
        ped_cells_nodes = tuple(map(tuple, ped_cells_index))
        ped_cells_nodes = {node:None for node in ped_cells_nodes}

        graph = copy.deepcopy(env.graph)
        cost = {}
        for edge in graph.edges():
            if edge[0] in ped_cells_nodes and edge[1] in ped_cells_nodes:
                cost[edge] = np.inf
        nx.set_edge_attributes(graph, cost, "cost")

        path = nx.astar_path(graph, env.robot_pos_map, env.goal_pos_map, heuristic=dist, weight="cost")
        path_cost = nx.path_weight(graph, path, 'cost')
        path_found = True
        if path_cost >= np.inf:
            path_found = False
        return path, path_found
    
    def get_lidar(self, env, first_episode=None, global_map=None):
        robot_pos = env.robot_pos_map
        yaw = env.robot_orientation_radian
        fov_angle = 128
        fov = math.radians(fov_angle)
        max_distance = 50

        #visible_cells = np.ones_like(env.occupancy_grid) * 0.5
        x = robot_pos[1]
        y = robot_pos[0]
        lidar_measurements = [max_distance*env.resolution] * fov_angle
        #l = [5] * 128
        for i, angle in enumerate(np.linspace(-fov/2, fov/2, int(fov_angle))):
            relative_angle = yaw + angle
            dx = math.cos(relative_angle)
            dy = math.sin(relative_angle)
            for r in np.linspace(0, max_distance, int(max_distance)):  # stepping in small increments
                ix = int(x + dx * r)
                iy = int(y + dy * r)
                
                if ix < 0 or iy < 0 or ix >= env.occupancy_grid.shape[1] or iy >= env.occupancy_grid.shape[0]:
                    # Out of bounds
                    break

                if env.occupancy_grid[iy][ix] == 0:
                    # Hit an obstacle
                    lidar_measurements[i] = dist(robot_pos, (iy,ix)) * env.resolution
                    break
        return lidar_measurements, None, None, None
    
    def get_local_map_raycast(self, env, first_episode=None, global_map=None):
        robot_pos = env.robot_pos_map
        yaw = env.robot_orientation_radian
        fov_angle = self.args.fov
        fov = math.radians(fov_angle)
        max_distance = 50

        # Add pedestrian position in the occupancy grid
        occupancy_grid = env.occupancy_grid.copy()
        if env.args.pedestrian_present:
            for ped in env.pedestrians:
                ped_pos_map = env.world_to_map(np.array(ped[0]))
                occupancy_grid = cv2.circle(occupancy_grid, ped_pos_map[::-1], int(env.orca_radius//env.resolution), 0, -1)

        # Raycast to get the visible cells
        visible_cells = np.ones_like(occupancy_grid) * 0.5
        for angle in np.linspace(-fov/2, fov/2, int(1.5*fov_angle)):
            ray_casting(visible_cells, robot_pos[1], robot_pos[0], yaw + angle, occupancy_grid, max_distance)
        
        # Crop grid around the robot
        map_dim = 2 * max_distance

        t_r = robot_pos[0]-map_dim//2
        top_row = max(0, t_r)

        b_r = robot_pos[0]+map_dim//2
        bottom_row = min(env.occupancy_grid.shape[0], b_r)

        l_c = robot_pos[1]-map_dim//2
        left_col = max(0, l_c)

        r_c = robot_pos[1]+map_dim//2
        right_col = min(env.occupancy_grid.shape[1], r_c)

        map_cut = visible_cells[top_row:bottom_row, left_col:right_col]

        # Cropped size might not be (map_dim,map_dim). 
        # Overlay cropped grid on a grid of size (map_dim, map_dim) of unknown values
        map = np.ones((map_dim, map_dim)) * 0.5
        r = abs(t_r) if t_r < 0 else 0
        c = abs(l_c) if l_c < 0 else 0
        map[r: r + map_cut.shape[0], c: c + map_cut.shape[1]] = map_cut

        # Rotate grid
        rotated_grid = rotate(map, np.degrees(yaw), reshape=False, mode='constant', cval=0.5, prefilter=True)

        # Change all the cell values to either free, occupied or unknown
        rotated_grid[rotated_grid<=0.4] = 0
        rotated_grid[rotated_grid>=0.6] = 1
        uk_mask = np.logical_and(rotated_grid!=0, rotated_grid!=1)
        rotated_grid[uk_mask] = 0.5

        # Mark the visible cells for plotting
        occupancy_grid[visible_cells==1] = 2
        #occupancy_grid[visible_cells==0] = 3

        # Create pedestrian map and get the closest pedestrian (if pedestrian is visible then mark those cells)           
        if env.args.obs_pedestrian_map:
            p_map = np.ones_like(env.occupancy_grid, dtype=float)
            flag = False        
            env.closest_visible_pedestrian = None 
            env.closest_visible_ped_dist = np.inf

            for ped in env.pedestrians:
                ped_pos_map = env.world_to_map(np.array(ped[0]))

                if visible_cells[ped_pos_map[0], ped_pos_map[1]] == 0: # ped is visible to robot
                    flag = True # visible pedestrian found

                    # Get the rotated pedestrian position and mark it in the map
                    ped_dist = env.get_relative_pos(ped[0])
                    ped_angle = env.get_relative_orientation(ped[0])
                    rotated_ped_pos = np.array((env.robot_pos[0] + ped_dist*np.cos(ped_angle), env.robot_pos[1] + ped_dist*np.sin(ped_angle)))
                    rotated_ped_pos_map = env.world_to_map(rotated_ped_pos)
                    p_map = cv2.circle(p_map, rotated_ped_pos_map[::-1], int(env.orca_radius/env.resolution), 0, -1)

                    if ped_dist < env.closest_visible_ped_dist:
                        env.closest_visible_ped_dist = ped_dist
                        env.closest_visible_pedestrian = ped
                    #pedestrian_map = cv2.circle(pedestrian_map, ped_pos_map[::-1], 2, 1, -1)
                    #pedestrian_map_ = cv2.circle(pedestrian_map_, ped_pos_map[::-1], 2, 0, -1)
                    
            # Get a (map_dim, map_dim) pedestrian map        
            p_map_cut = p_map[top_row:bottom_row, left_col:right_col]
            pedestrian_map = np.ones((map_dim, map_dim))
            pedestrian_map[r: r + p_map_cut.shape[0], c: c + p_map_cut.shape[1]] = p_map_cut
        else:
            pedestrian_map = None

        path_found = None
        if self.args.replan_if_collision:
            replan_map = env.inflated_grid.copy() 
            replan_needed = False   
            unavoidable_collision_distance_to_waypoint = self.args.pedestrian_collision_threshold #- self.args.waypoint_reach_threshold

            # Check if any visible pedestrian is within unavoidable_collision_distance_to_ first 3 waypoints
            for ped in env.pedestrians:
                ped_pos_map = env.world_to_map(np.array(ped[0]))
                if visible_cells[ped_pos_map[0], ped_pos_map[1]] == 0: #ped is visible
                    for waypoint in env.waypoints[:1]:                 
                        if dist(ped[0], waypoint) <= unavoidable_collision_distance_to_waypoint+.2:
                            replan_needed = True
                            break
                    if replan_needed:
                        break
            if replan_needed:
                # Mark visible cells in the inflated grid
                for ped in env.pedestrians:
                    ped_pos_map = env.world_to_map(np.array(ped[0]))
                    if visible_cells[ped_pos_map[0], ped_pos_map[1]] == 0: #ped is visible
                        replan_map = cv2.circle(replan_map, ped_pos_map[::-1], int((unavoidable_collision_distance_to_waypoint)/env.resolution), 0, -1)

                # Make the robot current position in the inflated grid to be free space.
                robot_pos_map = np.zeros(replan_map.shape)
                robot_pos_map = cv2.circle(robot_pos_map, robot_pos[::-1], math.ceil(self.args.inflation_radius), 1, -1)
                robot_pos_map = np.logical_and(robot_pos_map==1, env.occupancy_grid==1)
                #if np.any(robot_pos_map):
                #    env.visualize_map()
                replan_map[robot_pos_map] = 1

                path, path_found = self.plan(env, replan_map)
                #env.visualize_map()
                if path_found:
                    point_interval = 10 #changed
                    p = path[::point_interval][1:]

                    if tuple(env.goal_pos_map) not in p:
                        p.append(tuple(env.goal_pos_map))

                    env.waypoints = list(map(env.map_to_world, map(np.array, p)))
                    env.goal_pos = env.waypoints[-1]
                    env.visualize_map()
                else:
                    pass

        # Create pedestrian map (if pedestrian is visible then mark those cells)  
        #pedestrian_map = np.zeros_like(env.occupancy_grid, dtype=float)  
        #pedestrian_map_ = np.ones_like(env.occupancy_grid, dtype=float) 
        '''
        p_map = np.ones(env.occupancy_grid.shape)
        if env.args.obs_pedestrian_map:
            flag = False        
            env.closest_visible_pedestrian = None 
            env.closest_visible_ped_dist = np.inf
            for ped in env.pedestrians:
                ped_pos_map = env.world_to_map(np.array(ped[0]))

                if visible_cells[ped_pos_map[0], ped_pos_map[1]] == 0: # ped is visible to robot
                    try:
                        flag = True
                    
                        ped_dist = env.get_relative_pos(ped[0])
                        ped_angle = env.get_relative_orientation(ped[0])
                        rotated_ped_pos = np.array((env.robot_pos[0] + ped_dist*np.cos(ped_angle), env.robot_pos[1] + ped_dist*np.sin(ped_angle)))
                        rotated_ped_pos_map = env.world_to_map(rotated_ped_pos)
                        p_map = cv2.circle(p_map, rotated_ped_pos_map[::-1], int(env.orca_radius//env.resolution), 0, -1)

                        if ped_dist < env.closest_visible_ped_dist:
                            env.closest_visible_ped_dist = ped_dist
                            env.closest_visible_pedestrian = ped
                        #pedestrian_map = cv2.circle(pedestrian_map, ped_pos_map[::-1], 2, 1, -1)
                        #pedestrian_map_ = cv2.circle(pedestrian_map_, ped_pos_map[::-1], 2, 0, -1)
                    except:
                        print(ped_pos_map[::-1])
                    
            #pedestrian_map = cv2.circle(pedestrian_map, env.world_to_map(self.p)[::-1], 2, 1, -1) # fixed pedestrian
            p_map_cut = p_map[top_row:bottom_row, left_col:right_col]
            # ped_map_cut = pedestrian_map[top_row:bottom_row, left_col:right_col]
            # ped_map_cut_ = pedestrian_map_[top_row:bottom_row, left_col:right_col]

            # ped_map = np.zeros((map_dim, map_dim))
            # ped_map_ = np.zeros((map_dim, map_dim))
            p_map = np.zeros((map_dim, map_dim))

            # r = abs(t_r) if t_r < 0 else 0
            # c = abs(l_c) if l_c < 0 else 0
            # ped_map[r: r + ped_map_cut.shape[0], c: c + ped_map_cut.shape[1]] = ped_map_cut
            # ped_map_[r: r + ped_map_cut_.shape[0], c: c + ped_map_cut_.shape[1]] = ped_map_cut_


            p_map[r: r + p_map_cut.shape[0], c: c + p_map_cut.shape[1]] = p_map_cut
            

            # Rotate grid
            # ped_rotated_grid = rotate(ped_map, np.degrees(yaw), reshape=False, mode='constant', cval=0, prefilter=True)
            # ped_rotated_grid_ = rotate(ped_map_, np.degrees(yaw), reshape=False, mode='constant', cval=0, prefilter=True)

            # ped_rotated_grid[ped_rotated_grid<0.2] = 0
            # ped_rotated_grid[ped_rotated_grid>=0.2] = 1

            # ped_rotated_grid_[ped_rotated_grid_<0.2] = 0
            # ped_rotated_grid_[ped_rotated_grid_>=0.2] = 1
            
            pedestrian_map = p_map
    
        
        m = np.zeros_like(occupancy_grid, dtype=float)
        m[occupancy_grid == 1] = 0.5
        m[visible_cells == 1] = 1
        m[visible_cells == 0] = 0

        if flag:
            plt.figure()
            plt.imshow(rotated_grid)

            #plt.figure()
            #plt.imshow(ped_rotated_grid_)

            plt.figure()
            plt.imshow(p_map)

            plt.figure()
            plt.imshow(m)
            plt.plot(robot_pos[1], robot_pos[0],marker="o", markersize=2, alpha=0.8)
            plt.show()
        
        '''
        return rotated_grid, pedestrian_map, path_found, occupancy_grid
    
    def get_simple_local_map(self, env, first_episode=None, global_map=None):  
        # Cut (map_dim x map_dim) matrix from the occupancy grid centered around the robot
        map_dim = 100 # 100 x 100   
        robot_pos = env.robot_pos_map
        yaw = env.robot_orientation_radian

        t_r = robot_pos[0]-map_dim//2
        top_row = max(0, t_r)

        b_r = robot_pos[0]+map_dim//2
        bottom_row = min(env.occupancy_grid.shape[0], b_r)

        l_c = robot_pos[1]-map_dim//2
        left_col = max(0, l_c)

        r_c = robot_pos[1]+map_dim//2
        right_col = min(env.occupancy_grid.shape[1], r_c)

        
        occupancy_grid = env.occupancy_grid.copy()    
        if env.args.pedestrian_present:
            for ped in env.pedestrians:
                ped_pos_map = env.world_to_map(np.array(ped[0]))
                try:
                    occupancy_grid = cv2.circle(occupancy_grid, ped_pos_map[::-1], 5, 0, -1)
                except:
                    print(ped_pos_map[::-1])
        
        map_cut = occupancy_grid[top_row:bottom_row, left_col:right_col]
        
        if global_map:
            if first_episode == True:
                self.prev_global_map = None
            else:
                self.prev_global_map = self.global_map
            self.global_map[top_row:bottom_row, left_col:right_col] = map_cut

        # Overlap the partial map on a (map_dim x map_dim) zero np array
        partial_map = np.ones((map_dim, map_dim)) * 0.5
        #map = np.zeros((map_dim, map_dim)) 
        r = abs(t_r) if t_r < 0 else 0
        c = abs(l_c) if l_c < 0 else 0
        partial_map[r: r + map_cut.shape[0], c: c + map_cut.shape[1]] = map_cut        
        
        # Roate the occupancy grid by the robot's orientation (East facing)
        rotated_grid = rotate(partial_map, np.degrees(yaw), reshape=False, mode='constant', cval=0.5, prefilter=True)
        
        rotated_grid[rotated_grid<=0.4] = 0
        rotated_grid[rotated_grid>=0.6] = 1
        uk_mask = np.logical_and(rotated_grid!=0, rotated_grid!=1)
        rotated_grid[uk_mask] = 0.5

        #rotated_grid = rotate(map, np.degrees(env.robot_orientation_radian), reshape=True, mode='nearest')
        # Rotated grid might be larger than 100x100. So make it 100x100 centered around the robot
        # row_top = rotated_grid.shape[0]//2 - map_dim//2
        # row_bottom = rotated_grid.shape[0]//2 + map_dim//2
        # col_left = rotated_grid.shape[1]//2 - map_dim//2
        # col_right = rotated_grid.shape[1]//2 + map_dim//2
        # rotated_grid = rotated_grid[row_top: row_bottom, col_left:col_right]

        # Create pedestrian map (if pedestrian is visible then mark those cells)   
        if env.args.obs_pedestrian_map:
            p_map = np.ones_like(env.occupancy_grid, dtype=float)
            flag = False        
            env.closest_visible_pedestrian = None 
            env.closest_visible_ped_dist = np.inf
            for ped in env.pedestrians:
                ped_pos_map = env.world_to_map(np.array(ped[0]))

                if top_row <= ped_pos_map[0] < bottom_row and left_col <= ped_pos_map[1] < right_col: #ped is visible
                    try:
                        flag = True
                    
                        ped_dist = env.get_relative_pos(ped[0])
                        ped_angle = env.get_relative_orientation(ped[0])
                        rotated_ped_pos = np.array((env.robot_pos[0] + ped_dist*np.cos(ped_angle), env.robot_pos[1] + ped_dist*np.sin(ped_angle)))
                        rotated_ped_pos_map = env.world_to_map(rotated_ped_pos)
                        p_map = cv2.circle(p_map, rotated_ped_pos_map[::-1], int(env.orca_radius/env.resolution), 0, -1)

                        if ped_dist < env.closest_visible_ped_dist:
                            env.closest_visible_ped_dist = ped_dist
                            env.closest_visible_pedestrian = ped
                        #pedestrian_map = cv2.circle(pedestrian_map, ped_pos_map[::-1], 2, 1, -1)
                        #pedestrian_map_ = cv2.circle(pedestrian_map_, ped_pos_map[::-1], 2, 0, -1)
                    except:
                        print(ped_pos_map[::-1])
            p_map_cut = p_map[top_row:bottom_row, left_col:right_col]
            pedestrian_map = np.ones((map_dim, map_dim))
            pedestrian_map[r: r + p_map_cut.shape[0], c: c + p_map_cut.shape[1]] = p_map_cut
        else:
            pedestrian_map = None

        path_found = None
        if self.args.replan_if_collision:
            replan_map = env.inflated_grid.copy() 
            replan_needed = False   
            unavoidable_collision_distance_to_waypoint = self.args.pedestrian_collision_threshold #- self.args.waypoint_reach_threshold

            # Check if any visible pedestrian is within unavoidable_collision_distance_to_ first 3 waypoints
            for ped in env.pedestrians:
                ped_pos_map = env.world_to_map(np.array(ped[0]))
                if top_row <= ped_pos_map[0] < bottom_row and left_col <= ped_pos_map[1] < right_col: #ped is visible
                    for waypoint in env.waypoints[:3]:                 
                        if dist(ped[0], waypoint) <= unavoidable_collision_distance_to_waypoint:
                            replan_needed = True
                            break
                    if replan_needed:
                        break
            if replan_needed:
                # Mark visible cells in the inflated grid
                for ped in env.pedestrians:
                    ped_pos_map = env.world_to_map(np.array(ped[0]))
                    if top_row <= ped_pos_map[0] < bottom_row and left_col <= ped_pos_map[1] < right_col: #ped is visible
                        replan_map = cv2.circle(replan_map, ped_pos_map[::-1], int((unavoidable_collision_distance_to_waypoint+0.2)/env.resolution), 0, -1)

                # Make the robot current position in the inflated grid to be free space.
                robot_pos_map = np.zeros(replan_map.shape)
                robot_pos_map = cv2.circle(robot_pos_map, robot_pos[::-1], math.ceil(self.args.inflation_radius), 1, -1)
                robot_pos_map = np.logical_and(robot_pos_map==1, env.occupancy_grid==1)
                replan_map[robot_pos_map] = 1

                path, path_found = self.plan(env, replan_map)
                #env.visualize_map()
                if path_found:
                    point_interval = 10 #changed
                    p = path[::point_interval][1:]

                    if tuple(env.goal_pos_map) not in p:
                        p.append(tuple(env.goal_pos_map))

                    env.waypoints = list(map(env.map_to_world, map(np.array, p)))
                    env.goal_pos = env.waypoints[-1]
                    #env.visualize_map()
                else:
                    pass
        # Plot grid
        
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(env.occupancy_grid, cmap='gray')
        # plt.xlabel("orientation" + str(-env.robot_orientation_degree))
        # plt.plot(robot_pos[1], robot_pos[0], marker='o')
        # plt.subplot(1,2,2)
        # plt.imshow(rotated_grid)
        # plt.show()
        
        occupancy_grid[top_row:bottom_row, left_col:right_col][map_cut==1] = 2
        #occupancy_grid[top_row:bottom_row, left_col:right_col][map_cut==0] = 3
        return rotated_grid, pedestrian_map, path_found, occupancy_grid

    def get_observation(self, env, obs_encoder, mode="polar", action=None, first_episode=None, global_map=None, replan=None):
        # Observation: relative robot position and orientation
        if mode == "cartesian":
            # TO-DO
            task_obs = torch.tensor(env.task.target_pos[:-1] - env.robots[0].get_position()[:-1])
        else:
            relative_goal_pos = env.get_relative_pos(env.goal_pos)
            relative_goal_orientation = env.get_relative_orientation(env.goal_pos)
            task_obs = torch.tensor([relative_goal_pos, relative_goal_orientation])
            task_obs = task_obs.to(self.args.device).float().unsqueeze(0)
        
        # Observation: Map
        if self.args.env_type == "with_map":
            local_map, pedestrian_map, path_found, occupancy_grid = self.get_local_map(env, first_episode, global_map)
            local_map = torch.tensor(local_map)
            local_map = local_map.to(self.args.device).float()
        else:
            local_map = None

        
        # Observation: Waypoints
        closest_frontier_found = None
        if self.args.frontier_based_navigation:
            if replan == True:
                #frontiers = get_frontiers(self.global_map, frontier_size=3)
                #show_frontiers(self.global_map, frontiers)

                # If global map changes, create a new graph for path planning
                #if np.array_equal(self.global_map, self.prev_global_map) == False or \
                #    (first_episode==None and self.global_map_inflated_grid[env.robot_pos_map[0]][env.robot_pos_map[1]]!=1):
                    
                self.global_map_inflated_grid = inflate_grid(self.global_map, self.args.inflation_radius, 0, 0)
                
                # inflated grid might make the robot pos or goal pos to be not free. In that case
                # replace inflated grid cells with global map cells
                for i in range(env.robot_pos_map[0]-5, env.robot_pos_map[0]+6):
                    for j in range(env.robot_pos_map[1]-5, env.robot_pos_map[1]+6,):
                        self.global_map_inflated_grid[i][j] = self.global_map[i][j]
                for i in range(env.goal_pos_map[0]-5, env.goal_pos_map[0]+6):
                    for j in range(env.goal_pos_map[1]-5, env.goal_pos_map[1]+6,):
                        self.global_map_inflated_grid[i][j] = self.global_map[i][j]

                #plt.imshow(inflate_grid(env.occupancy_grid, self.args.inflation_radius, 0, 0));plt.show()

                # Create graph for path planning
                self.global_map_graph = nx.grid_graph((self.global_map_inflated_grid.shape[0], self.global_map_inflated_grid.shape[1]))
                nx.set_edge_attributes(self.global_map_graph, np.inf, "cost")

                free_space = np.vstack((np.where(self.global_map_inflated_grid==1)[0], np.where(self.global_map_inflated_grid==1)[1])).T
                free_space_nodes = tuple(map(tuple, free_space))
                free_space_nodes = {node:None for node in free_space_nodes}

                self.global_map_cost = {}
                for edge in self.global_map_graph.edges():
                    if edge[0] in free_space_nodes and edge[1] in free_space_nodes:
                        self.global_map_cost[edge] = np.linalg.norm(np.array(edge[0])-np.array(edge[1]))
                nx.set_edge_attributes(self.global_map_graph, self.global_map_cost, "cost")
                
                # get frontiers in the inflated grid
                frontiers = get_frontiers(self.global_map_inflated_grid, frontier_size=3)
                #show_frontiers(self.global_map_inflated_grid, frontiers)

                if self.args.frontier_selection_method == "closest":
                    # If goal in free space
                    closest_frontier_found = False
                    if self.global_map_inflated_grid[env.goal_pos_map[0]][env.goal_pos_map[1]] == 1:
                        closest_frontier_waypoints = nx.astar_path(self.global_map_graph, env.robot_pos_map, env.goal_pos_map, heuristic=dist, weight="cost")
                        #visualize_path(self.global_map_inflated_grid, closest_frontier_waypoints, show=True)
                        #visualize_path(self.global_map, closest_frontier_waypoints, show=True)
                        path_to_frontier_cost = 0

                        for i in range(len(closest_frontier_waypoints)-2):
                            edge_cost = self.global_map_graph[closest_frontier_waypoints[i]][closest_frontier_waypoints[i+1]]["cost"]
                            if edge_cost == np.inf:
                                path_to_frontier_cost = np.inf
                                break
                                plt.figure()
                                plt.imshow(self.global_map_inflated_grid)
                                plt.plot(path_to_frontier[i][1], path_to_frontier[i][0], marker='o')
                                plt.plot(path_to_frontier[i+1][1], path_to_frontier[i+1][0], marker='o')
                            path_to_frontier_cost += self.global_map_graph[closest_frontier_waypoints[i]][closest_frontier_waypoints[i+1]]["cost"]
                        #print(path_to_frontier_cost)
                        if path_to_frontier_cost != np.inf:
                            closest_frontier_found = True

                    # If goal not in free space found so far or there is no path from robot to goal
                    if self.global_map_inflated_grid[env.goal_pos_map[0]][env.goal_pos_map[1]] != 1 or closest_frontier_found==False:
                        closest_frontier = None
                        closest_frontier_found = False
                        closest_frontier_waypoints = None
                        closest_frontier_path_length = np.inf
                        for f in frontiers:
                            path_to_frontier = nx.astar_path(self.global_map_graph, env.robot_pos_map, tuple(f.centroid), heuristic=dist, weight="cost")
                            #path_to_frontier_cost = path_weight(self.global_map_graph, path_to_frontier[::-1], "cost")
                            path_to_frontier_cost = frontier_path_cost(self.global_map_graph, path_to_frontier)
                            
                            #visualize_path(self.global_map_inflated_grid, path_to_frontier, show=True)
                            #print(path_to_frontier_cost)
                            if path_to_frontier_cost >= np.inf:
                                continue

                            path_frontier_to_goal = nx.astar_path(self.all_free_map_graph, tuple(f.centroid), env.goal_pos_map, heuristic=dist)
                            path_frontier_to_goal_lenth = dist(f.centroid, env.goal_pos_map)

                            if path_to_frontier_cost + path_frontier_to_goal_lenth < closest_frontier_path_length:
                                closest_frontier_found = True
                                closest_frontier = f
                                closest_frontier_waypoints = path_to_frontier + path_frontier_to_goal
                        print("FRONTIER")
                        #visualize_path(self.global_map_inflated_grid, closest_frontier_waypoints, show=True) 
                if closest_frontier_found == False:
                    print("CLOSEST FRONTIER NOT FOUND")
                else:
                    env.waypoints = closest_frontier_waypoints
                    point_interval = 10 #changed
                    p = closest_frontier_waypoints[::point_interval][1:] # First point is robot start pos
                    
                    if tuple(env.goal_pos_map) not in p:
                        p.append(tuple(env.goal_pos_map))
                    
                    env.waypoints = list(map(env.map_to_world, map(np.array, p)))
                    env.goal_pos = env.waypoints[-1]
                
                #visualize_path(self.global_map_inflated_grid, closest_frontier_waypoints, show=True)
            else:
                closest_frontier_found = True

        # Get fixed number of waypoints from the full waypoints list
        waypoints = env.waypoints[:self.args.num_wps_input]
        #if replan==True:
            #visualize_path(self.global_map_inflated_grid, p, show=True) 
        if len(waypoints) == 0:
            waypoints = [env.goal_pos] # Needed for next_step_obs(in replay buffer) in the case of goal reached to pass without error 
        # Always have a fixed number of waypoint for NN input.
        while len(waypoints) < self.args.num_wps_input:
            waypoints.append(waypoints[-1])
        # Get relative waypoints
        if mode == "cartesian":
            waypoints -= env.robots[0].get_position()[:-1]
        else:
            # waypoints in polar coordinate in robot frame
            waypoints_rf = [np.array([env.get_relative_pos(p), env.get_relative_orientation(p)]) for p in waypoints] 

        waypoints = torch.tensor(np.array(waypoints_rf))
        waypoints = waypoints.reshape((-1,))
        waypoints_obs = waypoints.to(self.args.device).float().unsqueeze(0)

        # Observation: previous action
        if self.args.obs_previous_action:
            action = torch.tensor(action)
            action = action.to(self.args.device).float().unsqueeze(0)
        else:
            action = None

        # Observation: pedestrian map
        if self.args.obs_pedestrian_map:
            pedestrian_map = torch.tensor(pedestrian_map)
            pedestrian_map = pedestrian_map.to(self.args.device).float()
        else:
            pedestrian_map = None

        # Encode all observations
        with torch.no_grad():
            if self.args.obs_map_lstm and first_episode == True:
                obs_encoder.map_encoder.initialize()
            encoded_obs = obs_encoder(task_obs, waypoints_obs, local_map, action, pedestrian_map) 
        encoded_obs = encoded_obs.squeeze(0)
        return encoded_obs.detach().cpu().numpy(), path_found, waypoints_rf, closest_frontier_found, occupancy_grid

    def make_directory(self, scene, ep):
        # Create directory to save plots
        if self.args.pedestrian_present:
            folder_name = "pedestrian_present"
        else:
            folder_name = self.args.env_type
        path = f'output_simple_env/{folder_name}/{scene}_{self.args.checkpoint_name}_seed{self.args.train_seed}_test'
        if not os.path.exists(path):
            os.mkdir(path) 
        path = f'output_simple_env/{folder_name}/{scene}_{self.args.checkpoint_name}_seed{self.args.train_seed}_test/{ep}'  
        if not os.path.exists(path):
            os.mkdir(path)  
        return path  

    def submit(self, agent, args):
        #__________________________________Initialization__________________________________#
        self.args = copy.deepcopy(args)
        env_config = parse_config(self.config_file)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Initialize agent, encoder, writer and replay buffer
        low = np.array([-1,-1])
        high = np.array([1,1])
        action_space = gym.spaces.Box(low, high, dtype=np.float32)

        agent = SAC(num_inputs=256, action_space=action_space, args=args)
        if self.args.load_checkpoint == True:
            agent.load_checkpoint(ckpt_path=self.args.checkpoint_path, evaluate=True)
            
        obs_encoder = ObsEncoder(args).to(self.args.device)

        if args.map == "cropped_map":
            self.get_local_map = self.get_simple_local_map
        elif args.map == "raycast":
            self.get_local_map = self.get_local_map_raycast
        elif args.map == "lidar":
            self.get_local_map = self.get_lidar

        arrow = u'$\u2191$'
        
        if self.args.write_results:
            if self.args.pedestrian_present:
                folder_name = "pedestrian_present"
            else:
                folder_name = self.args.env_type
            result_file = open(f'output_simple_env/{folder_name}/{self.args.checkpoint_name}_seed{self.args.train_seed}_test.txt', 'w')
            
        for scene in self.test_scenes:           
            env = Simple_env_original(args, scene_id = scene)
            self.global_map = np.ones(env.occupancy_grid.shape) * 0.5

            if args.env_type == 'without_map':
                waypoints_map = np.ones((int((env.x_range[1]-env.x_range[0])/env.resolution), int((env.y_range[1]-env.y_range[0])/env.resolution)))
            elif args.env_type == 'with_map':
                waypoints_map = env.occupancy_grid
            
            total_numsteps = 0
            total_num_episodes = 0
            metrics = {key: 0.0 for key in [
            'success', 'episode_return', 'success_timestep', 'personal_space_violation_step', 'pedestrian_collision']}

            if self.args.write_results:
                result_file.write(f"SCENE: {scene}\n")
            
            #____________________________________EVALUATION START_________________________#
            for ep in range(1,self.eval_episodes_per_scene+1):
                print(f"scene: {scene} episode {ep}")
                args.train_seed += 1 # This makes every episode's start state to be same across multiple runs

                #if ep <6: continue
                # Plot for failed episodes
                # if ep in [60]:
                #    self.args.plot = True
                # else:
                #     continue
                #    self.args.plot = False
                    
                #if ep not in [6, 30, 41, 57, 129, 165, 167, 193]:
                #    continue

                env = Simple_env_original(args, scene_id = scene)             
                env.initialize_episode()

                # if ep == 2:
                #     #fixed_ped = env.orca_sim.addAgent((0, 0))
                #     p = env.waypoints[len(env.waypoints)//2]
                #     self.p = (p[0]-0.4, p[1]-0.1)
                    
                #     env.occupancy_grid = cv2.circle(env.occupancy_grid, env.world_to_map(self.p)[::-1], 2, 0, -1)
                #     plt.figure()
                #     plt.imshow(env.occupancy_grid);plt.show()
                #     self.args.plot = True
                # else:
                #     continue

                if self.args.frontier_based_navigation:
                    self.global_map = np.ones(env.occupancy_grid.shape) * 0.5
                    self.all_free_map = np.ones(env.occupancy_grid.shape)
                    self.all_free_map_graph = nx.grid_graph((self.all_free_map.shape[0], self.all_free_map.shape[1]))
                    nx.set_edge_attributes(self.all_free_map_graph, 1, "cost")
                    create_global_map = True
                    replan = False
                else:
                    create_global_map = False
                
                episode_reward = 0
                episode_steps = 0
                done = False

                

                if self.args.plot:
                    # Create directories to save plots
                    path = self.make_directory(scene, f'{ep}')

                #______________________________EPISODE START________________________#         
                while not done:
                    #print(episode_steps)
                    if self.args.frontier_based_navigation and episode_steps % self.args.replan_steps == 0 or len(env.waypoints) == 0:
                        replan = True
                    else:
                        replan = False

                    # GET OBSERVATION
                    if episode_steps == 0:
                        action = [0,0]
                        episode_start = True
                    else:
                        episode_start = False
                    
                    obs, path_found, waypoints, closest_frontier_found, occupancy_grid = self.get_observation(env, obs_encoder, mode="polar", action=action, first_episode = episode_start, global_map = create_global_map, replan=replan)
                    if self.args.plot:
                        plt.figure(figsize=(12,12))
                        # for plotting robot heading
                        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
                        if self.args.env_type == "without_map":
                            marker_rotation_angle = env.robot_orientation_degree-90
                            rotation_plot_degree = env.robot_orientation_degree
                            rotation_plot_radian = env.robot_orientation_radian
                            waypoints_angle_degree = np.degrees(waypoints[0][1])
                            waypoints_angle_radian = waypoints[0][1]
                        elif self.args.env_type == "with_map":
                            marker_rotation_angle = -env.robot_orientation_degree-90
                            rotation_plot_degree = -env.robot_orientation_degree
                            rotation_plot_radian = -env.robot_orientation_radian
                            waypoints_angle_degree = -np.degrees(waypoints[0][1])
                            waypoints_angle_radian = -waypoints[0][1]
                        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(marker_rotation_angle)

                        # Plot waypoints and robot pos
                        if self.args.env_type == "with_map":
                            waypoints_map = occupancy_grid
                            waypoints_map = cv2.circle(waypoints_map, env.robot_pos_map[::-1], int(env.orca_radius*10), 0, -1)
                        
                        plt.imshow(waypoints_map, cmap='gray')
                        waypoints_in_map = list(map(env.world_to_map, env.waypoints))
                        for waypoint in waypoints_in_map:
                            plt.plot(waypoint[1], waypoint[0], marker='o')
                        plt.scatter((env.robot_pos_map[1]), (env.robot_pos_map[0]), marker=rotated_marker, facecolors='none', edgecolors='b', s = 50)

                        # Plot pedestrians
                        if self.args.pedestrian_present:
                            colors = ['r','g','y', 'c', 'm', 'y', 'navy']
                            ped_distances = []
                            for i, ped in enumerate(env.pedestrians):
                                ped_pos = env.world_to_map(np.array(ped[0]))
                                ped_distances.append((colors[i], round(dist(ped[0], env.robot_pos), 2)))
                                # plot pedestrian 
                                rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
                                marker_rotation_angle = -np.degrees(ped[1])-90
                                rotated_marker._transform = rotated_marker.get_transform().rotate_deg(marker_rotation_angle)
                                plt.scatter(ped_pos[1], ped_pos[0], marker=rotated_marker, facecolors='none', edgecolors=colors[i], s=50)
                                
                                # plot pedestrian next target
                                ped_waypoints = env.pedestrian_waypoints[i]
                                if len(ped_waypoints) != 0:
                                    ped_target = env.world_to_map(np.array(ped_waypoints[0]))
                                    plt.plot(ped_target[1], ped_target[0], marker="*", color=colors[i])
                                
                                # plot pedestrian final target
                                if len(ped_waypoints) != 0:
                                    ped_target = env.world_to_map(np.array(ped_waypoints[-1]))
                                    plt.plot(ped_target[1], ped_target[0], marker="X", markersize=5, color=colors[i])

                        label = f'episode = {ep} step = {episode_steps}\n' + \
                            f'robot pos: {env.robot_pos} orientation: {rotation_plot_radian} (rad) {rotation_plot_degree} (degree) \n' + \
                            f'next waypoint pos: {env.waypoints[0]}\n' + \
                            f'next waypoint distance = {waypoints[0][0]}' + \
                                f'next waypoints angle = {waypoints_angle_radian}(rad) {waypoints_angle_degree} (degree)\n' + \
                                f'ped distances: {ped_distances} \n' 
                    
                    # TAKE A STEP
                    if env.no_of_collisions >= 5:
                        action = env.action_space.sample()
                    elif self.args.replan_if_collision and path_found == False:
                        action = [0, 0]
                    elif replan == True and closest_frontier_found == False:
                        action = [0, 0]
                    else:
                        action = agent.select_action(obs, evaluate=True)  # Sample action from policy
                    reward, reward_type, done, info = env.step(action) # Step

                    episode_steps += 1
                    total_numsteps += 1
                    episode_reward += reward

                    if self.args.plot:   
                        label += f'action = {action}' + \
                                    f'reward = {reward} total reward = {episode_reward}'
                        plt.xlabel(label)
                        plt.savefig(f'{path}/{episode_steps}.png')
                        plt.close()
                        
                total_num_episodes += 1               

                # Write results in file        
                if self.args.write_results:
                    result_file.write(f'ep {ep}\n success: {info["success"]}\n episode return: {episode_reward}\n steps: {episode_steps}\n')
                    if self.args.pedestrian_present:
                        result_file.write(f' collision with pedestrian: {info["pedestrian_collision"]}\n personal_space_violation_steps: {env.personal_space_violation_steps / episode_steps}\n')
                    
                    # if not info['success']:    
                    #     if info['pedestrian_collision']:
                    #         failed_episode_file.write(f'{ep} pedestrian_collision\n')
                    #     else:
                    #         failed_episode_file.write(f'{ep}\n')
                    # else:
                    #     failed_episode_file.write(f'{ep}\n')
                        

                if info['success']:
                    metrics['success_timestep'] += episode_steps
                else:
                    print("failed episode")
                    if self.args.pedestrian_present:
                        if info['pedestrian_collision']:
                            print('pedestrian collision')
                            #metrics['pedestrian_collision'] += 1
                metrics['episode_return'] += episode_reward
                metrics['personal_space_violation_step'] += env.personal_space_violation_steps
                for key in metrics:
                    if key in info:
                        metrics[key] += info[key]
                
            
            if self.args.write_results:
                result_file.write("--------------------------------------\n")
                result_file.write(f"SCENE {scene}\n")
                result_file.write(f'success rate: {metrics["success"]/ total_num_episodes}\n')
                result_file.write(f'Average reward: {metrics["episode_return"]/total_num_episodes}\n')
                result_file.write(f'success weighted by time steps: {metrics["success_timestep"]/metrics["success"]}\n')
                if self.args.pedestrian_present:
                    result_file.write(f'Personal space violation steps: {metrics["personal_space_violation_step"]/total_numsteps}\n')
                    result_file.write(f'no of episode with pedestrian collision: {metrics["pedestrian_collision"]}\n')
                result_file.write('---------------------------------------\n')
                
        result_file.close()
        
                        
if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
