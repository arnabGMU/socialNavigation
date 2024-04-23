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
import time

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
        self.eval_episodes_per_scene = 200
    
    def plan(self, env, ped_inflated_grid, visible_pedestrians=None):
        path_found = False
        slack = 0.2
        if len(env.waypoints) == 1: 
            return None, path_found
        # else:
        #     unavoidable_collision_distance_to_waypoint = self.args.pedestrian_collision_threshold
        #     point_found = False
        #     for wp in env.waypoints[1:2]:
        #         possible_collision = False
        #         for ped in visible_pedestrians:
        #             if dist(ped[0], wp) <= unavoidable_collision_distance_to_waypoint + slack:
        #                 possible_collision = True
        #         if possible_collision == False:
        #             point_found = True
        #             plan_point = env.world_to_map(wp)
        #             break
        # if point_found == False:            
        #     return None, path_found
        inflated_grid_new = copy.deepcopy(env.inflated_grid_new)

        # Get the cells that are occupied by visible pedestrians
        ped_cells = np.logical_and(env.inflated_grid==1, ped_inflated_grid==0)
        ped_cells_index = np.vstack((np.where(ped_cells==1)[0], np.where(ped_cells==1)[1])).T 
        ped_cells_nodes = tuple(map(tuple, ped_cells_index))
        ped_cells_nodes = {node:None for node in ped_cells_nodes}
        # Mark the cells as occupied by pedestrians in inflated grid
        for ped_cell in ped_cells_nodes.keys():
            inflated_grid_new.remove_node(ped_cell)
            # if (ped_cell[0], ped_cell[1]+1) in ped_cells_nodes:
            #     inflated_grid_new.remove_edge(ped_cell, (ped_cell[0], ped_cell[1]+1))
            # if (ped_cell[0]+1, ped_cell[1]) in ped_cells_nodes:
            #     inflated_grid_new.remove_edge(ped_cell, (ped_cell[0]+1, ped_cell[1]))
            # if (ped_cell[0]+1, ped_cell[1]+1) in ped_cells_nodes:
            #     inflated_grid_new.remove_edge(ped_cell, (ped_cell[0]+1, ped_cell[1]+1)) 
            
        # Get the robot current position cells in the grid
        robot_pos_map = np.zeros(env.occupancy_grid.shape)
        robot_pos_map = cv2.circle(robot_pos_map, env.robot_pos_map, math.ceil(self.args.inflation_radius), 1, -1)
        robot_pos_map = np.logical_and(robot_pos_map==1, env.occupancy_grid==1)
        robot_cells_index = np.vstack((np.where(robot_pos_map==1)[0], np.where(robot_pos_map==1)[1])).T 
        robot_cells_nodes = tuple(map(tuple, robot_cells_index))
        robot_cells_nodes = {node:None for node in robot_cells_nodes}

        # Robot current position might be occupied in the inflated grid. Before path planning mark
        # the robot current position as free in the inflated grid 
        for robot_cell in robot_cells_nodes.keys():
            if (robot_cell[0], robot_cell[1]+1) in robot_cells_nodes:
                inflated_grid_new.add_edge(robot_cell, (robot_cell[0], robot_cell[1]+1), weight=1)
            if (robot_cell[0]+1, robot_cell[1]) in robot_cells_nodes:
                inflated_grid_new.add_edge(robot_cell, (robot_cell[0]+1, robot_cell[1]), weight=1)
            if (robot_cell[0]+1, robot_cell[1]+1) in robot_cells_nodes:
                inflated_grid_new.add_edge(robot_cell, (robot_cell[0]+1, robot_cell[1]+1), weight=1.41)  
            if (robot_cell[0]-1, robot_cell[1]-1) in robot_cells_nodes:
                inflated_grid_new.add_edge(robot_cell, (robot_cell[0]-1, robot_cell[1]-1), weight=1.41)       

        try:
            #path = nx.shortest_path(inflated_grid_new, env.robot_pos_map, env.goal_pos_map)   
            path = nx.astar_path(inflated_grid_new, env.robot_pos_map, env.goal_pos_map, heuristic=dist, weight="weight") 
            path_found = True
            # plt.figure()
            # plt.imshow(ped_inflated_grid)
            # for p in path:
            #     plt.plot(p[1],p[0],marker='o', markersize=1)
            # plt.show()
        except:
            path = None
            path_found = False
            #print("path not found")

        # ped_cells = np.logical_and(env.inflated_grid==1, ped_inflated_grid==0)
        # ped_cells_index = np.vstack((np.where(ped_cells==1)[0], np.where(ped_cells==1)[1])).T 
        # ped_cells_nodes = tuple(map(tuple, ped_cells_index))
        # ped_cells_nodes = {node:None for node in ped_cells_nodes}

        # robot_pos_map = np.zeros(env.occupancy_grid.shape)
        # robot_pos_map = cv2.circle(robot_pos_map, env.robot_pos_map, math.ceil(self.args.inflation_radius), 1, -1)
        # robot_pos_map = np.logical_and(robot_pos_map==1, env.occupancy_grid==1)
        # robot_cells_index = np.vstack((np.where(robot_pos_map==1)[0], np.where(robot_pos_map==1)[1])).T 
        # robot_cells_nodes = tuple(map(tuple, robot_cells_index))
        # robot_cells_nodes = {node:None for node in robot_cells_nodes}

        # graph = copy.deepcopy(env.graph)
        # cost = {}
        # for edge in graph.edges():
        #     if edge[0] in ped_cells_nodes or edge[1] in ped_cells_nodes:
        #         cost[edge] = np.inf
        #     elif edge[0] in robot_cells_nodes and edge[1] in robot_cells_nodes:
        #         cost[edge] = dist(edge[0], edge[1])
        # nx.set_edge_attributes(graph, cost, "cost")

        # path = nx.astar_path(graph, env.robot_pos_map, plan_point, heuristic=dist, weight="cost")

        #env.visualize_map()
        
        # plt.figure()
        # plt.imshow(ped_inflated_grid)
        # for wp in env.waypoints:
        #     wp = env.world_to_map(wp)
        #     plt.plot(wp[1], wp[0], marker='o')
        # plt.show()

        # plt.figure()
        
        # plt.imshow(ped_inflated_grid)
        # c = 0
        # for p in path:
        #     plt.plot(p[1],p[0],marker='o', markersize=1)
        # plt.show()
        # path_cost = nx.path_weight(graph, path, 'cost')
    
        # path_found = True
        # if path_cost >= np.inf:
        #     path_found = False
        # else:
        #     plt.figure()
        #     plt.imshow(ped_inflated_grid)
        #     for p in path:
        #         plt.plot(p[1],p[0],marker='o', markersize=1)
        #     plt.show()
        return path, path_found

    # def plan(self, env, ped_inflated_grid):
    #     ped_cells = np.logical_and(env.inflated_grid==1, ped_inflated_grid==0)
    #     ped_cells_index = np.vstack((np.where(ped_cells==1)[0], np.where(ped_cells==1)[1])).T 
    #     ped_cells_nodes = tuple(map(tuple, ped_cells_index))
    #     ped_cells_nodes = {node:None for node in ped_cells_nodes}

    #     graph = copy.deepcopy(env.graph)
    #     cost = {}
    #     for edge in graph.edges():
    #         if edge[0] in ped_cells_nodes and edge[1] in ped_cells_nodes:
    #             cost[edge] = np.inf
    #     nx.set_edge_attributes(graph, cost, "cost")

    #     path = nx.astar_path(graph, env.robot_pos_map, env.goal_pos_map, heuristic=dist, weight="cost")
    #     path_cost = nx.path_weight(graph, path, 'cost')
    #     path_found = True
    #     if path_cost >= np.inf:
    #         path_found = False
    #     return path, path_found
    
    def get_local_map(self, env):
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

        '''
        plt.figure()
        plt.imshow(occupancy_grid)

        plt.figure()
        plt.imshow(visible_cells)

        plt.figure()
        plt.imshow(map)

        plt.figure()
        plt.imshow(rotated_grid);plt.show()
        '''

        # Change all the cell values to either free, occupied or unknown
        rotated_grid[rotated_grid<=0.4] = 0
        rotated_grid[rotated_grid>=0.6] = 1
        uk_mask = np.logical_and(rotated_grid!=0, rotated_grid!=1)
        rotated_grid[uk_mask] = 0.5

        # Mark the visible cells for plotting
        occupancy_grid[visible_cells==1] = 2
        #occupancy_grid[visible_cells==0] = 3

        # Create pedestrian map and get the closest pedestrian (if pedestrian is visible then mark those cells)   
        p_map = np.ones_like(env.occupancy_grid, dtype=float)
        if env.args.obs_pedestrian_map:
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
                        replan_map = cv2.circle(replan_map, ped_pos_map[::-1], int((unavoidable_collision_distance_to_waypoint + 0.2)/env.resolution), 0, -1)

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
        # r = abs(t_r) if t_r < 0 else 0
        # c = abs(l_c) if l_c < 0 else 0
        r = map_dim//2 - (robot_pos[0]-top_row)
        c = map_dim//2 - (robot_pos[1]-l_c)
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
            #p_map = np.ones_like(env.occupancy_grid, dtype=float)
            p_map = np.ones((map_dim, map_dim))
            flag = False        
            env.closest_visible_pedestrian = None 
            env.closest_visible_ped_dist = np.inf
            visible_pedestrians = []

            for ped in env.pedestrians:
                ped_pos_map = env.world_to_map(np.array(ped[0]))

                if top_row <= ped_pos_map[0] < bottom_row and left_col <= ped_pos_map[1] < right_col: #ped is visible
                    
                    visible_pedestrians.append(ped)
                
                    ped_dist = env.get_relative_pos(ped[0])
                    ped_angle = env.get_relative_orientation(ped[0])
                    #rotated_ped_pos = np.array((env.robot_pos[0] + ped_dist*np.cos(ped_angle), env.robot_pos[1] + ped_dist*np.sin(ped_angle)))
                    rotated_ped_pos_map = np.array((map_dim//2 + int((ped_dist*np.sin(ped_angle))/env.resolution), map_dim//2 + int((ped_dist*np.cos(ped_angle))/0.1)))
                    #rotated_ped_pos_map = env.world_to_map(rotated_ped_pos)
                    p_map = cv2.circle(p_map, rotated_ped_pos_map[::-1], int(env.orca_radius/env.resolution), 0, -1)

                    if ped_dist < env.closest_visible_ped_dist:
                        env.closest_visible_ped_dist = ped_dist
                        env.closest_visible_pedestrian = ped
                    
            # p_map_cut = p_map[top_row:bottom_row, left_col:right_col]
            # pedestrian_map = np.ones((map_dim, map_dim))
            # pedestrian_map[r: r + p_map_cut.shape[0], c: c + p_map_cut.shape[1]] = p_map_cut
            pedestrian_map = p_map
        else:
            pedestrian_map = None

        # if env.args.obs_pedestrian_map:
        #     p_map = np.ones_like(env.occupancy_grid, dtype=float)
        #     flag = False        
        #     env.closest_visible_pedestrian = None 
        #     env.closest_visible_ped_dist = np.inf
        #     visible_pedestrians = []

        #     for ped in env.pedestrians:
        #         ped_pos_map = env.world_to_map(np.array(ped[0]))

        #         if top_row <= ped_pos_map[0] < bottom_row and left_col <= ped_pos_map[1] < right_col: #ped is visible
        #             flag = True
        #             visible_pedestrians.append(ped)

        #             ped_dist = env.get_relative_pos(ped[0])
        #             ped_angle = env.get_relative_orientation(ped[0])
        #             rotated_ped_pos = np.array((env.robot_pos[0] + ped_dist*np.cos(ped_angle), env.robot_pos[1] + ped_dist*np.sin(ped_angle)))
        #             rotated_ped_pos_map = env.world_to_map(rotated_ped_pos)
        #             p_map = cv2.circle(p_map, rotated_ped_pos_map[::-1], int(env.orca_radius/env.resolution), 0, -1)

        #             if ped_dist < env.closest_visible_ped_dist:
        #                 env.closest_visible_ped_dist = ped_dist
        #                 env.closest_visible_pedestrian = ped
        #     p_map_cut = p_map[top_row:bottom_row, left_col:right_col]
        #     pedestrian_map = np.ones((map_dim, map_dim))
        #     pedestrian_map[r: r + p_map_cut.shape[0], c: c + p_map_cut.shape[1]] = p_map_cut
        # else:
        #     pedestrian_map = None

        path_found = None
        if self.args.replan_if_collision:
            replan_map = env.inflated_grid.copy() 
            replan_needed = False   
            unavoidable_collision_distance_to_waypoint = self.args.pedestrian_collision_threshold #- self.args.waypoint_reach_threshold

            # Check if any visible pedestrian is within unavoidable_collision_distance_to_ first waypoints
            for ped in visible_pedestrians:
                if dist(ped[0], env.waypoints[0]) <= unavoidable_collision_distance_to_waypoint + 0.2:
                    replan_needed = True
                    break

            # # Check if any visible pedestrian is within unavoidable_collision_distance_to_ first 3 waypoints
            # for ped in env.pedestrians:
            #     ped_pos_map = env.world_to_map(np.array(ped[0]))
            #     if top_row <= ped_pos_map[0] < bottom_row and left_col <= ped_pos_map[1] < right_col: #ped is visible
            #         for waypoint in env.waypoints[:1]:                 
            #             if dist(ped[0], waypoint) <= unavoidable_collision_distance_to_waypoint + 0.2:
            #                 replan_needed = True
            #                 break
            #         if replan_needed:
            #             break
            if replan_needed:
                # Mark visible cells in the inflated grid
                for ped in visible_pedestrians:
                    ped_pos_map = env.world_to_map(np.array(ped[0]))
                    replan_map = cv2.circle(replan_map, ped_pos_map[::-1], int((unavoidable_collision_distance_to_waypoint + 0.2)/env.resolution), 0, -1)
                env.replan = True

                # # Mark visible cells in the inflated grid
                # for ped in env.pedestrians:
                #     ped_pos_map = env.world_to_map(np.array(ped[0]))
                #     if top_row <= ped_pos_map[0] < bottom_row and left_col <= ped_pos_map[1] < right_col: #ped is visible
                #         replan_map = cv2.circle(replan_map, ped_pos_map[::-1], int((unavoidable_collision_distance_to_waypoint+0.2)/env.resolution), 0, -1)

                # Make the robot current position in the inflated grid to be free space.
                # robot_pos_map = np.zeros(replan_map.shape)
                # robot_pos_map = cv2.circle(robot_pos_map, robot_pos[::-1], math.ceil(self.args.inflation_radius), 1, -1)
                # robot_pos_map = np.logical_and(robot_pos_map==1, env.occupancy_grid==1)
                # replan_map[robot_pos_map] = 1

                path, path_found = self.plan(env, replan_map, visible_pedestrians)
                #env.visualize_map()
                if path_found:
                    point_interval = 10 #changed
                    p = path[::point_interval][1:]
                    p = path[int(len(path)/2)]
                    p = np.array(env.map_to_world(np.array(p)))
                    
                    #env.ghost_node = np.array(env.map_to_world(p))

                    # env.visualize_map()
                    # env.waypoints[0] = np.array(p)
                    # env.visualize_map()

                    #if tuple(env.goal_pos_map) not in p:
                    #    p.append(tuple(env.goal_pos_map))

                    #env.waypoints = list(map(env.map_to_world, map(np.array, p)))
                    #env.goal_pos = env.waypoints[-1]
                    #env.visualize_map()
                else:
                    env.ghost_node = None
            else:
                env.replan = False
                env.ghost_node = None
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
            #local_map, pedestrian_map, path_found, occupancy_grid = self.get_simple_local_map(env, first_episode, global_map)
            local_map, pedestrian_map, path_found, occupancy_grid = self.get_local_map(env)
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

        # Observation: replan
        if self.args.obs_replan:
            if env.replan == True:
                replan = torch.tensor([1])
            else:
                replan = torch.tensor([0])
            replan = replan.to(self.args.device).float().unsqueeze(0)
        else:
            replan = None

        # Encode all observations
        with torch.no_grad():
            if self.args.obs_map_lstm and first_episode == True:
                obs_encoder.map_encoder.initialize()
            encoded_obs = obs_encoder(task_obs, waypoints_obs, local_map, action, pedestrian_map, replan) 
        encoded_obs = encoded_obs.squeeze(0)
        return encoded_obs.detach().cpu().numpy(), path_found, waypoints_rf, closest_frontier_found, occupancy_grid

    def make_directory(self, scene, ep):
        # Create directory to save plots
        if self.args.pedestrian_present:
            folder_name = "pedestrian_present"
        else:
            folder_name = self.args.env_type
        path = f'output_simple_env/{folder_name}/{scene}_{self.args.checkpoint_name}_seed{self.args.train_seed}'
        if not os.path.exists(path):
            os.mkdir(path) 
        path = f'output_simple_env/{folder_name}/{scene}_{self.args.checkpoint_name}_seed{self.args.train_seed}/{ep}'  
        if not os.path.exists(path):
            os.mkdir(path)  
        return path  
    
    def get_local_map_dwa(self, map_dim):
        #map_dim = 100 # 100 x 100   
        robot_pos = self.env.robot_pos_map

        t_r = robot_pos[0]-map_dim//2
        top_row = max(0, t_r)

        b_r = robot_pos[0]+map_dim//2
        bottom_row = min(self.env.occupancy_grid.shape[0], b_r)

        l_c = robot_pos[1]-map_dim//2
        left_col = max(0, l_c)

        r_c = robot_pos[1]+map_dim//2
        right_col = min(self.env.occupancy_grid.shape[1], r_c)

        occupancy_grid = self.env.occupancy_grid.copy()
        if self.env.args.pedestrian_present:
            for ped in self.env.pedestrians:
                ped_pos_map = self.env.world_to_map(np.array(ped[0]))
                try:
                    occupancy_grid = cv2.circle(occupancy_grid, ped_pos_map[::-1], 5, 0, -1)
                except:
                    print(ped_pos_map[::-1])
        local_map = np.ones_like(occupancy_grid)
        local_map[top_row:bottom_row, left_col:right_col] = occupancy_grid[top_row:bottom_row, left_col:right_col]

        return local_map
    
    def collision_next_step(self, x_init, v, w, local_map):
        x = np.array(x_init)
        for i in range(2):
            x = self.motion(x, [v,w], self.env.action_timestep)
            pos_map = self.env.world_to_map((x[0], x[1]))
            
            if local_map[pos_map[0]][pos_map[1]] == 0: # collision
                # print(x_init)
                # print(x)
                # print(pos_map)
                # plt.figure()
                # plt.imshow(local_map)
                # plt.plot(pos_map[1], pos_map[0], marker='o')
                # plt.show()
                # plt.close()
                return True
        return False
    
    def motion(self, x, u, dt):
        """
        motion model
        """
        
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x
    def predict_trajectory(self, x_init, v, y, local_map):
        """
        predict trajectory with an input
        """
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        predict_time = 1
        dt = self.env.action_timestep
        collision = False
        
        while time <= predict_time:
            x = self.motion(x, [v, y], dt)
            trajectory = np.vstack((trajectory, x))
            time += dt
            # # break if wall collision to save time
            trajectory_pos_map = self.env.world_to_map((x[0], x[1]))
            
            if local_map[trajectory_pos_map[0]][trajectory_pos_map[1]] == 0:
                collision = True
                break

        return trajectory, collision, time
    
    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        #assert abs(cost_angle) == cost, f'{cost_angle} {cost}'
        normalized_cost = self.normalize_obs(cost, min=0, max=math.pi, min_range=0)
        #normalized_cost = cost
        # print("gc", cost)
        # print(normalized_cost)
        return normalized_cost
    
    def calc_obstacle_cost(self, trajectory, local_map):
        """
        calc obstacle cost inf: collision
        """
        collision = False
        ox = np.where(local_map==0)[0]
        oy = np.where(local_map==0)[1]
        # no obstacle in local map
        if len(ox) == 0:
            return 0, collision

        trajectory = np.array(list(map(self.env.world_to_map, trajectory[:,:2])))
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy) * self.env.action_timestep

        
        if np.array(r <= self.args.pedestrian_collision_threshold).any():
            collision = True
            return None, collision
        else:
            min_r = np.min(r)
        normalized_cost = self.normalize_obs(1.0/min_r, min=1.0/(local_map.shape[0] * self.env.action_timestep), max = 1.0/self.args.pedestrian_collision_threshold, min_range=0)
        #return 1.0 / min_r  # OK
        # print("obs_cost", 1/min_r)
        # print(normalized_cost)
        return normalized_cost, collision

    def calc_control_and_trajectory(self, x, dw, goal, index=None):
        """
        calculation final input with dynamic window
        """
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])
        v_resolution = 0.05
        yaw_rate_resolution = 5 * math.pi / 180
        robot_stuck_flag_cons = 0.001
        local_map = self.get_local_map_dwa(map_dim=20)

        to_goal_cost_gain = 0.4
        speed_cost_gain = 1
        obstacle_cost_gain = 0.1
        # evaluate all trajectory with sampled input in dynamic window
        
        # ped_map = self.env.occupancy_grid.copy()
        # for ped in self.env.pedestrians:
        #     ped_pos_map = self.env.world_to_map(np.array(ped[0]))
        #     ped_map = cv2.circle(ped_map, ped_pos_map[::-1], int(self.env.orca_radius/self.env.resolution), 0, -1)

        # plt.figure(figsize=(12,12))
        # plt.subplot(1,3,1)
        # plt.imshow(ped_map, cmap='gray')
        # arrow = u'$\u2191$'
        # rotated_marker = mpl.markers.MarkerStyle(marker=arrow)        
        # marker_rotation_angle = -self.env.robot_orientation_degree-90
        # rotated_marker._transform = rotated_marker.get_transform().rotate_deg(marker_rotation_angle)
        # goal_map = self.env.world_to_map(goal)
        # plt.scatter((self.env.robot_pos_map[1]), (self.env.robot_pos_map[0]), marker=rotated_marker, facecolors='none', edgecolors='r', s = 50)
        # waypoints_in_map = list(map(self.env.world_to_map, self.env.waypoints))
        # for waypoint in waypoints_in_map:
        #     plt.plot(waypoint[1], waypoint[0], marker='o')

        # plt.subplot(1,3,2)
        # plt.imshow(ped_map, cmap='gray')
        # plt.scatter((self.env.robot_pos_map[1]), (self.env.robot_pos_map[0]), marker=rotated_marker, facecolors='none', edgecolors='r', s = 50)

        # t_start_ = time.time()
        for v in np.arange(dw[0], dw[1]+v_resolution, v_resolution):
            for y in np.arange(dw[2], dw[3]+yaw_rate_resolution, yaw_rate_resolution):
                # not admissible velocity
                # if self.collision_next_step(x_init, v, y, local_map):
                #     continue
                
                # t_start = time.time()
                trajectory, collision, time_to_obstacle = self.predict_trajectory(x_init, v, y, local_map)
                if collision:
                    continue
                # t_end = time.time()
                # print("predict trajectory", t_end-t_start)

                # trajectory_map = np.array(list(map(self.env.world_to_map, trajectory[:,:2]))) 
                # plt.plot(trajectory_map[:,1], trajectory_map[:,0], 'b--')
                                             
                # calc cost
                # t_start = time.time()
                to_goal_cost = self.calc_to_goal_cost(trajectory, goal)
                # t_end = time.time()
                # print("goal cost trajectory", t_end-t_start)
                # print("sc", self.env.robot_linear_velocity - trajectory[1, 3])
                speed_cost =  self.env.robot_linear_velocity - trajectory[1, 3]
                # t_start = time.time()
                ob_cost, collision = self.calc_obstacle_cost(trajectory, local_map)
                if collision:
                    continue
                # if collision:
                #     ob_cost = 1
                # else:
                #     #r = 1.5 # prediction time + 0.5
                #     ob_cost = obstacle_cost_gain * self.calc_obstacle_cost(trajectory, local_map)
                #ob_cost = obstacle_cost_gain * self.normalize_obs(1.0/r, min=1/1.5, max=1/0.2, min_range=0)
                #ob_cost = obstacle_cost_gain * self.calc_obstacle_cost(trajectory, local_map)
                # t_end = time.time()
                # print("obs cost trajectory", t_end-t_start)

                final_cost = to_goal_cost_gain * to_goal_cost + speed_cost_gain * speed_cost + obstacle_cost_gain * ob_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    
                    # best_trajectory_map = np.array(list(map(self.env.world_to_map, best_trajectory[:,:2]))) 
                    # str = f'"goal_cost" {to_goal_cost}\n'
                    # str += f'"sc" {speed_cost}\n'
                    # str += f'"obs_cost" {ob_cost}\n'
                    # str += f'"final_cost" {final_cost}\n'
                    # str += f'"linear and angular velocity" {best_u}\n'

                    if abs(best_u[0]) < robot_stuck_flag_cons \
                            and abs(x[3]) < robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -self.env.robot_angular_velocity
        if min_cost == float('inf'):
            best_u = [0,0]
        # t_end_ = time.time()
        # print("overall", t_end_-t_start_)
        # sys.exit()
        
        # plt.subplot(1,3,3)
        # ped_map = cv2.circle(ped_map, self.env.robot_pos_map[::-1], int(self.env.orca_radius/self.env.resolution), 0, -1)
        # plt.imshow(ped_map, cmap='gray')
                        
        # #plt.plot(self.env.robot_pos_map[1], self.env.robot_pos_map[0], marker = 'o')       
        # plt.plot(goal_map[1], goal_map[0], marker = 'o')
        # plt.plot(best_trajectory_map[:,1], best_trajectory_map[:,0], 'b--')   
        # plt.scatter((self.env.robot_pos_map[1]), (self.env.robot_pos_map[0]), marker=rotated_marker, facecolors='none', edgecolors='r', s = 50)     
        # plt.xlabel(str)
        # #plt.show()
        # path = self.make_directory('Beechwood_1_int', 1)
        # plt.savefig(f'{path}/{index}.png')
        # plt.close()
        #sys.exit()
        return best_u, best_trajectory
        
    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [-self.env.robot_linear_velocity, self.env.robot_linear_velocity,
                -self.env.robot_angular_velocity, self.env.robot_angular_velocity]
        dw = Vs

        # # Dynamic window from motion model
        # Vd = [x[3] - config.max_accel * config.dt,
        #         x[3] + config.max_accel * config.dt,
        #         x[4] - config.max_delta_yaw_rate * config.dt,
        #         x[4] + config.max_delta_yaw_rate * config.dt]

        # #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        # dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
        #         max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw
    
    def dwa_control(self, x, goal, index = None):
        """
        Dynamic Window Approach control
        """
        dw = self.calc_dynamic_window(x)

        u, trajectory = self.calc_control_and_trajectory(x, dw, goal, index)

        return u, trajectory
    
    def normalize_obs(self, obs, min, max, min_range=-1):
        if min_range == 0:
            norm = (obs-min)/(max-min)
            return np.clip(norm, 0, 1)
        else:
            norm = (2*((obs-min)/(max-min))) - 1
            return np.clip(norm, -1, 1)

    def submit(self, agent, args):
        #__________________________________Initialization__________________________________#
        self.args = copy.deepcopy(args)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        metrics = {key: 0.0 for key in [
            'success', 'episode_return', 'success_timestep', 'personal_space_violation_step', 'pedestrian_collision']}
        
        # Initialize agent, encoder, writer and replay buffer
        low = np.array([-1,-1])
        high = np.array([1,1])
        action_space = gym.spaces.Box(low, high, dtype=np.float32)

        # Select map type
        if args.map == "cropped_map":
            self.get_local_map = self.get_simple_local_map
        elif args.map == "raycast":
            self.get_local_map = self.get_local_map_raycast
        elif args.map == "lidar":
            self.get_local_map = self.get_lidar
        
        scene_id = ['Beechwood_1_int', 'Ihlen_0_int']
        scene = scene_id[0]
        #env = Simple_env(args, scene_id = 'Beechwood_1_int')
        env = Simple_env_original(args, scene_id = scene)
        self.env = env
        self.global_map = np.ones(env.occupancy_grid.shape) * 0.5

        if args.env_type == 'without_map':
            waypoints_map = np.ones((int((env.x_range[1]-env.x_range[0])/env.resolution), int((env.y_range[1]-env.y_range[0])/env.resolution)))
        elif args.env_type == 'with_map':
            waypoints_map = env.occupancy_grid

        if self.args.write_results:
            if self.args.pedestrian_present:
                folder_name = "pedestrian_present"
            else:
                folder_name = self.args.env_type
            result_file = open(f'output_simple_env/{folder_name}/{scene}_{self.args.checkpoint_name}_seed{self.args.train_seed}.txt', 'w')

        arrow = u'$\u2191$'
        
        #____________________________________EVALUATION START_________________________#
        total_numsteps = 0
        total_num_episodes = 0       
        total_time = 0
        for ep in range(1,self.args.eval_episodes_per_scene+1):
            print("episode", ep)
            if self.args.plot:
                # Create directories to save plots
                path = self.make_directory(scene, f'{ep}')
            args.train_seed += 1 # This makes every episode's start state to be same across multiple runs

            # if ep < 100:
            #     continue
            #if ep not in [6, 30, 41, 57, 129, 165, 167, 193]:
            #    continue

            env = Simple_env_original(args, scene_id = scene)             
            env.initialize_episode()
            self.env = env
            # if self.args.frontier_based_navigation:
            #     self.global_map = np.ones(env.occupancy_grid.shape) * 0.5
            #     self.all_free_map = np.ones(env.occupancy_grid.shape)
            #     self.all_free_map_graph = nx.grid_graph((self.all_free_map.shape[0], self.all_free_map.shape[1]))
            #     nx.set_edge_attributes(self.all_free_map_graph, 1, "cost")
            #     create_global_map = True
            #     replan = False
            # else:
            #     create_global_map = False
            
            episode_steps = 0
            done = False
            replan_map_last_step = None

            # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
            x = np.array([env.robot_pos[0], env.robot_pos[1], env.robot_orientation_radian, 0.0, 0.0])        
            trajectory = np.array(x)
            
            start_time = time.time()
            #env.visualize_map()
            #______________________________EPISODE START________________________#         
            while not done:
                # if self.args.frontier_based_navigation and episode_steps % self.args.replan_steps == 0 or len(env.waypoints) == 0:
                #     replan = True
                # else:
                #     replan = False

                # Get best linear and angular velocity from DWA
                
                u, predicted_trajectory = self.dwa_control(x, env.waypoints[0], index=episode_steps)
                u = [self.normalize_obs(u[0], min=-self.env.robot_linear_velocity, max=self.env.robot_linear_velocity),\
                     self.normalize_obs(u[1], min=-self.env.robot_angular_velocity, max=self.env.robot_angular_velocity)]
                
                # STEP
                reward, reward_type, done, info = env.step(u)

                

                # Add new position to trajectory
                x = np.array([env.robot_pos[0], env.robot_pos[1], env.robot_orientation_radian, 0.0, 0.0])        
                trajectory = np.vstack((trajectory, x))

                # replan if pedestrian is close to waypoints
                replan_needed = False
                for ped in env.pedestrians:
                    for wp in env.waypoints[:6]:
                        if dist(ped[0], wp) <= 0.5: # pedestrian is near waypoint 
                            replan_needed = True
                            replan_map = env.inflated_grid.copy() 
                            slack = 0.2
                            for ped in env.pedestrians:
                                ped_pos_map = env.world_to_map(np.array(ped[0]))
                                replan_map = cv2.circle(replan_map, ped_pos_map[::-1], int((0.3 + slack)/env.resolution), 0, -1)

                            if replan_map_last_step is not None and np.array(replan_map == replan_map_last_step).all():
                                path_found = False
                            else:
                                path, path_found = self.plan(env, replan_map, env.pedestrians)
                            #env.visualize_map()
                            if path_found:
                                # Update waypoints
                                point_interval = self.args.waypoint_interval #changed
                                if len(path) > point_interval:
                                    p = path[::point_interval][1:]
                                else:
                                    p = path
                                env.waypoints = list(map(env.map_to_world, map(np.array, p)))
                                env.goal_pos = env.waypoints[-1]
                                #env.visualize_map()
                                replan_map_last_step = None
                            else:
                                replan_map_last_step = replan_map
                            break
                    if replan_needed:
                        break


                if self.args.plot:
                    plt.figure(figsize=(12,12))
                    # for plotting robot heading
                    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
                    if self.args.env_type == "without_map":
                        marker_rotation_angle = env.robot_orientation_degree-90

                        rotation_plot_degree = env.robot_orientation_degree
                        rotation_plot_radian = env.robot_orientation_radian
                        waypoints_angle_degree = np.degrees(env.waypoints[0][1])
                        waypoints_angle_radian = env.waypoints[0][1]
                    elif self.args.env_type == "with_map":
                        marker_rotation_angle = -env.robot_orientation_degree-90

                        rotation_plot_degree = -env.robot_orientation_degree
                        rotation_plot_radian = -env.robot_orientation_radian
                        waypoints_angle_degree = -np.degrees(env.waypoints[0][1])
                        waypoints_angle_radian = -env.waypoints[0][1]
                    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(marker_rotation_angle)

                    # Plot waypoints robot pos
                    if self.args.env_type == "with_map":
                        waypoints_map = env.occupancy_grid.copy()
                        waypoints_map = cv2.circle(waypoints_map, env.robot_pos_map[::-1], int(env.orca_radius*10), 0, -1)

                    # plot pedestrians
                    for ped in env.pedestrians:
                        ped_pos = env.world_to_map(ped[0])
                        waypoints_map = cv2.circle(waypoints_map, ped_pos[::-1], int(env.orca_radius*10), 0, -1)
                    plt.imshow(waypoints_map, cmap='gray')

                    # plot waypoints
                    waypoints_in_map = list(map(env.world_to_map, env.waypoints))
                    for waypoint in waypoints_in_map:
                        plt.plot(waypoint[1], waypoint[0], marker='o')
                    plt.scatter((env.robot_pos_map[1]), (env.robot_pos_map[0]), marker=rotated_marker, facecolors='none', edgecolors='b', s = 50)

                    # Plot pedestrian heading and target position
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
                    
                    # plot predicted trajectory
                    trajectory_map = np.array(list(map(self.env.world_to_map, predicted_trajectory[:,:2])))
                    plt.plot(trajectory_map[:, 1], trajectory_map[:, 0], "b--")
                    label = f'episode = {ep} step = {episode_steps}\n' + \
                        f'robot pos: {env.robot_pos} orientation: {rotation_plot_radian} (rad) {rotation_plot_degree} (degree) \n' + \
                        f'next waypoint pos: {env.waypoints[0]}\n' + \
                        f'next waypoint distance = {env.waypoints[0][0]}' + \
                            f'next waypoints angle = {waypoints_angle_radian}(rad) {waypoints_angle_degree} (degree)\n' + \
                            f'ped distances: {ped_distances} \n' 
                
                # TAKE A STEP
                # if env.no_of_collisions >= 5:
                #     action = env.action_space.sample()
                # elif self.args.replan_if_collision and path_found == False:
                #     action = [0, 0]
                # # elif replan == True and closest_frontier_found == False:
                # #     action = [0, 0]
                # else:
                #     action = agent.select_action(obs, evaluate=True)  # Sample action from policy
                # reward, reward_type, done, info = env.step(action) # Step

                episode_steps += 1
                total_numsteps += 1

                if self.args.plot:   
                    label += f'action = {u}'
                    plt.xlabel(label)
                    plt.savefig(f'{path}/{episode_steps}.png')
                    plt.close()
                self.env = env

            end_time = time.time()  
            episode_time = end_time-start_time   
            total_time += episode_time
                
            total_num_episodes += 1               

            # Write results in file        
            if self.args.write_results:
                result_file.write(f'ep {ep}\n success: {info["success"]}\n steps: {episode_steps}\n')
                if self.args.pedestrian_present:
                    result_file.write(f' collision with pedestrian: {info["pedestrian_collision"]}\n personal_space_violation_steps: {env.personal_space_violation_steps / episode_steps}\n')                    

            if info['success']:
                metrics['success_timestep'] += episode_steps
            else:
                print("failed episode")
                if self.args.pedestrian_present:
                    if info['pedestrian_collision']:
                        print('pedestrian collision')
                        #metrics['pedestrian_collision'] += 1
            metrics['personal_space_violation_step'] += env.personal_space_violation_steps
            for key in metrics:
                if key in info:
                    metrics[key] += info[key]
            print("episode time", episode_time)
            
        print("average runtime per episode", total_time/total_num_episodes)
        if self.args.write_results:
            result_file.write(f'success rate: {metrics["success"]/ total_num_episodes}\n')
            result_file.write(f'Average reward: {metrics["episode_return"]/total_num_episodes}\n')
            result_file.write(f'success weighted by time steps: {metrics["success_timestep"]/metrics["success"]}\n')
            if self.args.pedestrian_present:
                result_file.write(f'Personal space violation steps: {metrics["personal_space_violation_step"]/total_numsteps}\n')
                result_file.write(f'no of episode with pedestrian collision: {metrics["pedestrian_collision"]}\n')
                result_file.write(f"average runtime per episode: {total_time/total_num_episodes}")
            result_file.write('---------------------------------------\n')
            result_file.close()
        
                        
if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
