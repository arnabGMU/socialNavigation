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
from occupancy_grid.occupancy_grid import get_lidar, get_local_map_raycast, get_simple_local_map
from gibson2.utils.constants import *
from encoder.obs_encoder import ObsEncoder
#from simple_env import Simple_env
from simple_env_original import Simple_env_original


logging.getLogger().setLevel(logging.WARNING)

#gc.set_threshold(0)

class Challenge:
    def __init__(self):
        self.config_file = CONFIG_FILE
        # self.training_scenes = ["Beechwood_1_int", "Ihlen_0_int", "Ihlen_1_int", "Merom_0_int"]
        # self.test_scenes = ["Benevolence_0_int", "Rs_int", "Wainscott_1_int"]
        # self.valdiation_scene = ["Pomaria_0_int"]
        self.training_scenes = ["Benevolence_0_int", "Ihlen_0_int", "Pomaria_0_int", "Wainscott_0_int", "Rs_int", "Merom_0_int", "Beechwood_0_int"]
        self.test_scenes = ["Beechwood_1_int", "Benevolence_2_int", "Ihlen_1_int", "Wainscott_1_int", "Merom_1_int", "Pomaria_2_int"]
        self.valdiation_scene = ["Pomaria_1_int", "Benevolence_1_int"]

    def get_observation(self, env, obs_encoder, action=None, first_episode=None, validation=None):
        relative_goal_pos = env.get_relative_pos(env.goal_pos)
        relative_goal_orientation = env.get_relative_orientation(env.goal_pos)
        if env.args.obs_normalized:
            relative_goal_pos = self.normalize_obs(relative_goal_pos, min=0.1, max=15)
            relative_goal_orientation = self.normalize_obs(relative_goal_orientation, min=-math.pi, max=math.pi)
        task_obs = torch.tensor([relative_goal_pos, relative_goal_orientation])
        task_obs = task_obs.to(env.args.device).float().unsqueeze(0)
        
        # Observation: Map
        if env.args.env_type == "with_map":
            local_map, pedestrian_map, path_found, visible_pedestrians, visible_cells, replan_map = self.get_local_map(env, validation)
            if self.args.obs_normalized:
                local_map = self.normalize_obs(local_map, min=0, max=1)
            local_map = torch.tensor(local_map)
            local_map = local_map.to(self.args.device).float()
        else:
            local_map = None

        # Observation: Waypoints
        # Get fixed number of waypoints from the full waypoints list
        no_of_waypoints = env.args.num_wps_input
        waypoints = env.waypoints[:no_of_waypoints]
        if len(waypoints) == 0:
            waypoints = [env.goal_pos] # Needed for next_step_obs(in replay buffer) in the case of goal reached to pass without error 
        # Always have a fixed number of waypoint for NN input.
        while len(waypoints) < no_of_waypoints:
            waypoints.append(waypoints[-1])
        # Get relative waypoints in polar coordinate in robot frame
        if self.args.obs_normalized:
            farthest_possible_wp_pos = no_of_waypoints*env.point_interval*env.resolution
            waypoints = [np.array([self.normalize_obs(env.get_relative_pos(p), min=0, max = farthest_possible_wp_pos), \
                                                        self.normalize_obs(env.get_relative_orientation(p), min=-math.pi, max=math.pi)]) for p in waypoints] 
        else:    
            waypoints = [np.array([env.get_relative_pos(p), env.get_relative_orientation(p)]) for p in waypoints] 
        
        waypoints = torch.tensor(np.array(waypoints))
        waypoints = waypoints.reshape((-1,))
        waypoints_obs = waypoints.to(self.args.device).float().unsqueeze(0)

        # Observation: previous action
        if self.args.obs_previous_action:
            if self.args.obs_normalized:
                action = [self.normalize_obs(action[0], min=-1, max=1), self.normalize_obs(action[1], min=-1, max=1)]
            action = torch.tensor(action)
            action = action.to(self.args.device).float().unsqueeze(0)
        else:
            action = None

        # Observation: pedestrian map
        if self.args.obs_pedestrian_map:
            if self.args.obs_normalized:
                pedestrian_map = self.normalize_obs(pedestrian_map, min=0, max=1)
            pedestrian_map = torch.tensor(pedestrian_map)
            pedestrian_map = pedestrian_map.to(self.args.device).float()
        else:
            pedestrian_map = None

        # Observation: replan
        # if self.args.obs_replan:
        #     if env.replan == True:
        #         replan = torch.tensor([1])
        #     else:
        #         replan = torch.tensor([0])
        #     replan = replan.to(self.args.device).float().unsqueeze(0)
        # else:
        #     replan = None

        # Observation: Pedestrian Pos 
        if self.args.obs_pedestrian_pos:
            ped_pos = []
            visible_pedestrians = sorted(visible_pedestrians, key=lambda x: env.get_relative_pos(x[0]))
            for ped in visible_pedestrians[:self.args.fixed_num_pedestrians]:
                ped_relative_pos_wc = env.get_relative_pos(ped[0])
                if ped_relative_pos_wc <= env.args.depth:
                    relative_ped_pos_wc_normalized = self.normalize_obs(ped_relative_pos_wc, min=0.3, max=env.args.depth)
                    relative_ped_orientation_normalized = self.normalize_obs(env.get_relative_orientation(ped[0]), min=-math.pi, max=math.pi)
                    relative_ped_heading = self.normalize_obs((ped[1]-env.robot_orientation_radian)%2*math.pi, min=-2*math.pi, max=2*math.pi)
                    ped_pos.append(np.array([relative_ped_pos_wc_normalized, relative_ped_orientation_normalized, relative_ped_heading]))
            if len(ped_pos) == 0:
                ped_pos.append(np.array([1,0, 0]))
            while len(ped_pos) < self.args.fixed_num_pedestrians:
                ped_pos.append(ped_pos[-1]) 
            
            ped_pos = torch.tensor(np.array(ped_pos))
            ped_pos = ped_pos.reshape((-1,))
            ped_pos_obs = ped_pos.to(self.args.device).float().unsqueeze(0)
        else:
            ped_pos_obs = None

        # Encode all observations
        if self.args.obs_train:
            # if self.args.obs_map_lstm and first_episode == True:
            #     obs_encoder.map_encoder.initialize()

            # print("task obs", task_obs)
            # print("waypoint", waypoints_obs)
            # print("map", local_map)
            # print("action", action)
            # print("pmap", pedestrian_map)
            encoded_obs = obs_encoder(task_obs, waypoints_obs, local_map, action, pedestrian_map, ped_pos_obs)
        else:
            with torch.no_grad():
                # if self.args.obs_map_lstm and first_episode == True:
                #     obs_encoder.map_encoder.initialize()
                # print("task obs", task_obs)
                # print("waypoint", waypoints_obs)
                # print("map", local_map)
                # print("action", action)
                # print("pmap", pedestrian_map)
                encoded_obs = obs_encoder(task_obs, waypoints_obs, local_map, action, pedestrian_map, ped_pos_obs) 
        encoded_obs = encoded_obs.squeeze(0)
        return encoded_obs.detach().cpu().numpy(), path_found, visible_cells, replan_map
    
    def normalize_obs(self, obs, min, max, min_range=-1):
        if min_range == 0:
            norm = (obs-min)/(max-min)
            return np.clip(norm, 0, 1)
        else:
            norm = (2*((obs-min)/(max-min))) - 1
            return np.clip(norm, -1, 1)

    def make_directory(self, scene, ep):
        # Create directory to save plots
        if self.args.pedestrian_present:
            folder_name = "pedestrian_present/plots"
        else:
            folder_name = self.args.env_type
        path = f'output_simple_env/{folder_name}/{scene}_{self.args.checkpoint_name}_seed{self.args.train_seed}'
        if not os.path.exists(path):
            os.mkdir(path) 
        path = f'output_simple_env/{folder_name}/{scene}_{self.args.checkpoint_name}_seed{self.args.train_seed}/{ep}'  
        if not os.path.exists(path):
            os.mkdir(path)  
        return path  

    def submit(self, agent, args):
        #__________________________________Initialization__________________________________#
        self.args = copy.deepcopy(args)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Initialize agent, encoder, writer and replay buffer
        low = np.array([-1,-1])
        high = np.array([1,1])
        action_space = gym.spaces.Box(low, high, dtype=np.float32)

        agent = SAC(num_inputs=256, action_space=action_space, args=args)
        obs_encoder = ObsEncoder(args).to(self.args.device)
        
        if self.args.load_checkpoint == True:
            agent.load_checkpoint(ckpt_path=self.args.checkpoint_path, evaluate=True)
            obs_encoder.load_checkpoint(ckpt_path=self.args.checkpoint_path, evaluate=True)
            
        if args.map == "cropped_map":
            self.get_local_map = get_simple_local_map
        elif args.map == "raycast":
            self.get_local_map = get_local_map_raycast
        elif args.map == "lidar":
            self.get_local_map = get_lidar

        arrow = u'$\u2191$'
        
        if self.args.write_results:
            if self.args.pedestrian_present:
                folder_name = "pedestrian_present"
            else:
                folder_name = self.args.env_type
            result_file = open(f'output_simple_env/{folder_name}/{self.args.checkpoint_name}_seed{self.args.train_seed}_test.txt', 'w')

        sr = 0
        time_step = 0
        personal_space_violation = 0
        ped_collision = 0
        timeout = 0
        total_steps = 0

        for scene in self.test_scenes:           
            env = Simple_env_original(args, scene_id = scene)
            self.global_map = np.ones(env.occupancy_grid.shape) * 0.5
            args.train_seed = args.eval_seed_start

            if args.env_type == 'without_map':
                waypoints_map = np.ones((int((env.x_range[1]-env.x_range[0])/env.resolution), int((env.y_range[1]-env.y_range[0])/env.resolution)))
            elif args.env_type == 'with_map':
                waypoints_map = env.occupancy_grid
            
            total_numsteps = 0
            total_num_episodes = 0
            metrics = {key: 0.0 for key in [
            'success', 'episode_return', 'success_timestep', 'personal_space_violation_step', 'pedestrian_collision', 'timeout']}

            if self.args.write_results:
                result_file.write(f"SCENE: {scene}\n")
            
            #____________________________________EVALUATION START_________________________#
            for ep in range(1, self.args.eval_episodes_per_scene+1):
                print(f"scene: {scene} episode {ep}")
                args.train_seed += 1 # This makes every episode's start state to be same across multiple runs
                    
                if ep not in [2]:
                    continue

                env = Simple_env_original(args, scene_id = scene)             
                env.initialize_episode()
                
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

                if self.args.plot_save:
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
                    
                    obs, path_found, visible_cells, replan_map = self.get_observation(env, obs_encoder, action=action, first_episode = episode_start)
                    if self.args.plot:
                        plt.figure(figsize=(12,12))
                        # for plotting robot heading
                        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
                        waypoint = env.waypoints[0]
                        waypoint_dist = env.get_relative_pos(waypoint)
                        waypoints_angle_radian = env.get_relative_orientation(waypoint)
                        waypoints_angle_degree = np.degrees(waypoints_angle_radian)
                        if self.args.env_type == "without_map":
                            marker_rotation_angle = env.robot_orientation_degree-90
                            rotation_plot_degree = env.robot_orientation_degree
                            rotation_plot_radian = env.robot_orientation_radian
                        elif self.args.env_type == "with_map":
                            marker_rotation_angle = -env.robot_orientation_degree-90
                            rotation_plot_degree = -env.robot_orientation_degree
                            rotation_plot_radian = -env.robot_orientation_radian
                            waypoints_angle_degree = -waypoints_angle_degree
                            waypoints_angle_radian = -waypoints_angle_radian
                        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(marker_rotation_angle)

                        # Plot waypoints and robot pos
                        if self.args.env_type == "with_map":
                            if replan_map is not None:
                                waypoints_map = replan_map
                                min_cell_value = np.min(waypoints_map) - 0.1
                                waypoints_map[waypoints_map==0] = min_cell_value
                            else:
                                waypoints_map = env.occupancy_grid.copy()
                            waypoints_map[visible_cells==1] = 2
                            waypoints_map = cv2.circle(waypoints_map, env.robot_pos_map[::-1], 1, 0, -1)
                        #waypoints_map[visible_cells==0] = 2
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
                                circle = plt.Circle(ped_pos[::-1], radius=3, color=colors[i])
                                plt.gca().add_patch(circle)
                                # plot pedestrian 
                                rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
                                marker_rotation_angle = -np.degrees(ped[1])-90
                                rotated_marker._transform = rotated_marker.get_transform().rotate_deg(marker_rotation_angle)
                                plt.scatter(ped_pos[1], ped_pos[0], marker=rotated_marker, facecolors='none', edgecolors=colors[i], s=50)
                                # try:
                                #     waypoints_map = cv2.circle(waypoints_map, ped_pos[::-1], 3, min_cell_value, -1)
                                # except:
                                #     waypoints_map = cv2.circle(waypoints_map, ped_pos[::-1], 3, 0, -1)
                                #     plt.figure()
                                #     plt.imshow(waypoints_map, cmap='gray')
                                #     plt.show()
                                
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
                            f'next waypoint distance = {waypoint_dist}' + \
                                f'next waypoints angle = {waypoints_angle_radian}(rad) {waypoints_angle_degree} (degree)\n' + \
                                f'ped distances: {ped_distances} \n' 
                    
                    # TAKE A STEP
                    if env.no_of_collisions >= 5:
                        action = env.action_space.sample()
                    else:
                        action = agent.select_action(obs, evaluate=True)  # Sample action from policy
                    reward, reward_type, done, info = env.step(action) # Step

                    episode_steps += 1
                    total_numsteps += 1
                    total_steps += 1
                    episode_reward += reward

                    if self.args.plot:   
                        label += f'action = {action}' + \
                                    f'reward = {reward} total reward = {episode_reward}'
                        plt.xlabel(label)
                        if self.args.plot_save:
                            plt.savefig(f'{path}/{episode_steps}.png')
                        else:
                            plt.show()
                        plt.close()
                        
                total_num_episodes += 1               

                # Write results in file        
                if self.args.write_results:
                    result_file.write(f'ep {ep}\n success: {info["success"]}\n episode return: {episode_reward}\n steps: {episode_steps}\n')
                    if self.args.pedestrian_present:
                        result_file.write(f' collision with pedestrian: {info["pedestrian_collision"]}\n personal_space_violation_steps: {env.personal_space_violation_steps / episode_steps}\n')
                        result_file.flush()
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
                
            sr += metrics["success"]
            time_step += metrics["success_timestep"]
            personal_space_violation += metrics["personal_space_violation_step"]
            ped_collision += metrics["pedestrian_collision"]
            timeout += metrics["timeout"]

            if self.args.write_results:
                result_file.write("--------------------------------------\n")
                result_file.write(f"SCENE {scene}\n")
                result_file.write(f'success rate: {metrics["success"]/ total_num_episodes}\n')
                result_file.write(f'Average reward: {metrics["episode_return"]/total_num_episodes}\n')
                result_file.write(f'Average time steps for successful episodes: {metrics["success_timestep"]/metrics["success"]}\n')
                if self.args.pedestrian_present:
                    result_file.write(f'Personal space violation steps: {metrics["personal_space_violation_step"]/total_numsteps}\n')
                    result_file.write(f'no of episode with pedestrian collision: {metrics["pedestrian_collision"]}\n')
                result_file.write(f'no of episode with timeout: {metrics["timeout"]}\n')
                result_file.write('---------------------------------------\n')
                result_file.flush()

        total_num_episodes = len(self.test_scenes) * self.args.eval_episodes_per_scene
        if self.args.write_results:
            result_file.write("--------------------------------------\n")
            result_file.write(f'Average success rate: {round((sr/total_num_episodes), 3)}\n')
            result_file.write(f'Average time steps for successful episodes: {time_step/sr}\n')
            if self.args.pedestrian_present:
                result_file.write(f'Personal space violation steps: {personal_space_violation/total_steps}\n')
                result_file.write(f'Average number of episodes with pedestrian collision: {ped_collision/total_num_episodes}\n')
            result_file.write(f'Average number of episodes with timeout: {timeout/total_num_episodes}\n')
            result_file.write('---------------------------------------\n')
            result_file.flush()
        result_file.close()
        
                        
if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
