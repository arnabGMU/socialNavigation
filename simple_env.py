import numpy as np
import random
import networkx as nx
from gibson2.utils.utils import dist
import gym
import math
from gibson2.utils.utils import l2_distance
import pickle
import os
from matplotlib import pyplot as plt
from occupancy_grid.occupancy_grid import visualize_path

class Simple_env:
    def __init__(self, args) -> None:
        self.x_range = (-15.0, 15.0)
        self.y_range = (-15.0, 15.0)
        self.resolution = 0.1
        self.num_wps_input = args.num_wps_input
        # Make action space as a gym box
        low = np.array([-1,-1])
        high = np.array([1,1])
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.robot_linear_velocity = 0.5
        self.robot_angular_velocity = 1.5707963267948966
        self.action_timestep = 0.1
        self.waypoints = None
        #random.seed(123) #train
        random.seed(1234) #evaluation

        # Build graph for pathfinding
        path = f'graph_simple_env_size{self.x_range[1]}.pickle'
        if not os.path.exists(path):
            og_x_size = int((self.x_range[1]-self.x_range[0])/self.resolution)
            og_y_size = int((self.y_range[1]-self.y_range[0])/self.resolution)
            self.graph = nx.grid_graph((og_x_size, og_y_size))
            cost = {}
            for edge in self.graph.edges():
                cost[edge] = np.linalg.norm(np.array(edge[0]) - np.array(edge[1]))
            nx.set_edge_attributes(self.graph, cost, "cost")
            pickle.dump(self.graph, open(path, 'wb'))
        else:
            self.graph = pickle.load(open(path, 'rb'))

    def initialize_episode(self):
        self.step_number = 0
        flag = True
        dist_threshold = 10
        # Take robot start and goal position in less than 10m
        while flag:
            # Initialize robot starting and goal point in world and map coordinate
            self.robot_pos = (random.uniform(self.x_range[0], self.x_range[1]),\
                            random.uniform(self.y_range[0], self.y_range[1]))
            self.robot_pos_map = self.world_to_map(self.robot_pos)
            self.robot_orientation_degree = random.uniform(-180, 180)
            self.robot_orientation_radian = np.radians(self.robot_orientation_degree)

            self.goal_pos = (random.uniform(self.x_range[0], self.x_range[1]),\
                            random.uniform(self.y_range[0], self.y_range[1]))
            self.goal_pos_map = self.world_to_map(self.goal_pos)

            if l2_distance(self.robot_pos, self.goal_pos) <= dist_threshold:
                flag = False
            else:
                continue
            
            # Compute path and generate waypoints
            path = nx.astar_path(self.graph, self.robot_pos_map, self.goal_pos_map, heuristic=dist, weight="cost")
            point_interval = 5
            p = path[::point_interval][1:]
            
            if tuple(self.goal_pos_map) not in p:
                p.append(tuple(self.goal_pos_map))
            
            self.waypoints = list(map(self.map_to_world, map(np.array, p)))
            self.goal_pos = self.waypoints[-1]
            #print("robot pos", self.robot_pos)
            #print("robot orientation", self.robot_orientation_degree)
            #print("goal pos", self.goal_pos)
            #print(self.waypoints)
            wayypints_map = np.ones((int((self.x_range[1]-self.x_range[0])/self.resolution), int((self.y_range[1]-self.y_range[0])/self.resolution)))
            #visualize_path(wayypints_map, p, show=True)

    
    def step(self, action):
        self.step_number += 1
        linear_velocity = action[0] * self.robot_linear_velocity
        angular_velocity = action[1] * self.robot_angular_velocity

        x = linear_velocity * np.cos(self.robot_orientation_radian) * self.action_timestep
        y = linear_velocity * np.sin(self.robot_orientation_radian) * self.action_timestep
        
        self.robot_pos = (self.robot_pos[0]+x, self.robot_pos[1]+y)
        self.robot_pos_map = self.world_to_map(self.robot_pos)
        
        self.robot_orientation_radian += (angular_velocity * self.action_timestep)
        self.robot_orientation_radian = self.robot_orientation_radian % (2*math.pi)
        self.robot_orientation_degree = np.degrees(self.robot_orientation_radian)

        reward = self.get_rewards()
        done, info = self.get_termination()

        return reward, done, info
    
    def get_termination(self):
        # goal termination condition
        info = {'success':0, 'episode_return':0}
        done = 0
        threshold = 0.1
        dist_to_goal = l2_distance(self.robot_pos, self.goal_pos)
        if dist_to_goal <= threshold:
            done = 1
            info['success'] = 1
        # max 500 step termination condition
        if self.step_number == 500:
            done =  1
        # out of bound
        if not(self.x_range[0] <= self.robot_pos[0] <= self.x_range[1] and\
            self.y_range[0] <= self.robot_pos[1] <= self.y_range[1]):
            done = 1
        return done, info

    def get_rewards(self):
        # Goal reward
        reward = 0.0
        reward += self.goal_reward()
        if reward > 0:
            return reward
        reward += self.orientation_reward()
        reward += self.potential_reward()
        reward += self.waypoint_reward()
        reward += self.timestep_reward()
        return reward
    
    def goal_reward(self):
        threshold = 0.1
        dist_to_goal = l2_distance(self.robot_pos, self.goal_pos)
        if dist_to_goal <= threshold:
            return 10
        return 0

    def orientation_reward(self):
        orientation_reward_weight = -0.01
        threshold = 0.1
        waypoint = self.waypoints[0]
        angle = abs(self.get_relative_orientation(waypoint))

        if l2_distance(self.robot_pos, waypoint) > threshold:
            return orientation_reward_weight * angle
        return 0

    def potential_reward(self):
        potential_reward_weight = -0.1
        threshold = 0.1
        waypoint = self.waypoints[0]
        dist = l2_distance(self.robot_pos, waypoint)
        if dist > threshold:
            return potential_reward_weight * dist
        return 0

    def waypoint_reward(self):
        waypoint_reward = 0.1
        threshold = 0.1
        waypoint = self.waypoints[0]
        if l2_distance(waypoint, self.robot_pos) <= threshold:
            self.waypoints.pop(0)
            return waypoint_reward
        return 0

    def timestep_reward(self):
        return -0.001

    def map_to_world(self, point):
        x = (point[0] * self.resolution) + self.x_range[0]
        y = (point[1] * self.resolution) + self.y_range[0]

        return (x,y)
    
    def world_to_map(self, point):
        x = int((point[0] - self.x_range[0]) / self.resolution)
        y = int((point[1] - self.y_range[0]) / self.resolution)

        return (x,y)
    
    def get_relative_pos(self, point):
        return np.linalg.norm(np.array(self.robot_pos) - np.array(point))
    
    def get_relative_orientation(self, point):
        p = np.array(point) - np.array(self.robot_pos)
        angle = np.arctan2(p[1], p[0])
        return angle - self.robot_orientation_radian


    
