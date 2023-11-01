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

from scipy.ndimage import rotate
from matplotlib import pyplot as plt
from PIL import Image

from gibson2.utils.utils import parse_config
from SAC.SAC import SAC
from occupancy_grid.occupancy_grid import visualize_path
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
    
    def get_simple_local_map(self, env):  
        # Cut (map_dim x map_dim) matrix from the occupancy grid centered around the robot
        map_dim = 100 # 100 x 100   
        robot_pos = env.robot_pos_map

        t_r = robot_pos[0]-map_dim//2
        top_row = max(0, t_r)

        b_r = robot_pos[0]+map_dim//2
        bottom_row = min(env.occupancy_grid.shape[0], b_r)

        l_c = robot_pos[1]-map_dim//2
        left_col = max(0, l_c)

        r_c = robot_pos[1]+map_dim//2
        right_col = min(env.occupancy_grid.shape[1], r_c)

        map_cut = env.occupancy_grid[top_row:bottom_row, left_col:right_col]

        # Overlap the partial map on a (map_dim x map_dim) zero np array
        map = np.zeros((map_dim, map_dim))
        r = abs(t_r) if t_r < 0 else 0
        c = abs(l_c) if l_c < 0 else 0
        map[r: r + map_cut.shape[0], c: c + map_cut.shape[1]] = map_cut        
       
        # Roate the occupancy grid by the robot's orientation (East facing)
        rotated_grid = rotate(map, np.degrees(env.robot_orientation_radian), reshape=True, mode='nearest')
        # Rotated grid might be larger than 100x100. So make it 100x100 centered around the robot
        row_top = rotated_grid.shape[0]//2 - map_dim//2
        row_bottom = rotated_grid.shape[0]//2 + map_dim//2
        col_left = rotated_grid.shape[1]//2 - map_dim//2
        col_right = rotated_grid.shape[1]//2 + map_dim//2
        rotated_grid = rotated_grid[row_top: row_bottom, col_left:col_right]

        # Plot grid
        '''
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(env.occupancy_grid, cmap='gray')
        plt.xlabel("orientation" + str(-env.robot_orientation_degree))
        plt.plot(robot_pos[1], robot_pos[0], marker='o')
        plt.subplot(1,2,2)
        plt.imshow(rotated_grid)
        plt.show()
        '''
        
        return rotated_grid

    def get_observation(self, env, obs_encoder, mode="polar", action=None):
        # Observation: relative robot position and orientation
        if mode == "cartesian":
            # TO-DO
            task_obs = torch.tensor(env.task.target_pos[:-1] - env.robots[0].get_position()[:-1])
        else:
            relative_goal_pos = env.get_relative_pos(env.goal_pos)
            relative_goal_orientation = env.get_relative_orientation(env.goal_pos)
            task_obs = torch.tensor([relative_goal_pos, relative_goal_orientation])
            task_obs = task_obs.to(self.args.device).float().unsqueeze(0)

        # Observation: Waypoints
        # Get fixed number of waypoints from the full waypoints list
        waypoints = env.waypoints[:self.args.num_wps_input]
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
        
        # Observation: Map
        if self.args.env_type == "with_map":
            map = self.get_simple_local_map(env)
            map = torch.tensor(map)
            map = map.to(self.args.device).float()
        else:
            map = None

        # Observation: previous action
        if self.args.obs_previous_action:
            action = torch.tensor(action)
            action = action.to(self.args.device).float().unsqueeze(0)
        else:
            action = None

        # Encode all observations
        with torch.no_grad():
            encoded_obs = obs_encoder(task_obs, waypoints_obs, map, action) 
        encoded_obs = encoded_obs.squeeze(0)
        return encoded_obs.detach().cpu().numpy(), waypoints_rf

    def submit(self, agent, args):
        self.args = args
        env_config = parse_config(self.config_file)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        task = env_config['task']
        if task == 'interactive_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'spl', 'effort_efficiency', 'ins', 'episode_return']}
        elif task == 'social_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'episode_return', 'success_timestep']}
        else:
            assert False, 'unknown task: {}'.format(task)
        
        # Initialize agent, encoder, writer and replay buffer
        low = np.array([-1,-1])
        high = np.array([1,1])
        action_space = gym.spaces.Box(low, high, dtype=np.float32)
        agent = SAC(num_inputs=256, action_space=action_space, args=args)
        if self.args.load_checkpoint == True:
            agent.load_checkpoint(ckpt_path=self.args.checkpoint_path, evaluate=True)
        obs_encoder = ObsEncoder(args).to(self.args.device)
        
        total_numsteps = 0
        total_num_episodes = 0
        scene_id = ['Beechwood_1_int', 'Ihlen_0_int']
        scene = scene_id[0]
        #env = Simple_env(args, scene_id = 'Beechwood_1_int')
        env = Simple_env_original(args, scene_id = scene)

        if args.env_type == 'without_map':
            wayypints_map = np.ones((int((env.x_range[1]-env.x_range[0])/env.resolution), int((env.y_range[1]-env.y_range[0])/env.resolution)))
        elif args.env_type == 'with_map':
            wayypints_map = env.occupancy_grid

        if self.args.write_results:
            result_file = open(f'output_simple_env/{scene}_{self.args.checkpoint_name}_seed{self.args.train_seed}.txt', 'w')
            failed_episode_file = open(f'output_simple_env/failed_episodes/{scene}_{self.args.checkpoint_name}_seed{self.args.train_seed}.txt', 'w')
        #failed_episodes = []
        #for i in failed_episode_file.readlines():
        #    failed_episodes.append(int(i))
        arrow = u'$\u2191$'
        
        for ep in range(1,self.eval_episodes_per_scene+1):
            print(ep)
        
            env.initialize_episode()
            #if ep != 10:
            #    continue
            
            #print(f'start pos: {env.robot_pos} goal pos: {env.goal_pos}')
            #print(f'orientation {(env.robot_orientation_degree)}')
            #print(f'orientation rad {env.robot_orientation_radian}')
            #print(f'waypoint {env.waypoints[0]}')
            
            episode_reward = 0
            episode_steps = 0
            done = False

            #if ep not in failed_episodes:
            #    continue

            if self.args.plot:
                path = f'output_simple_env/{scene}_{self.args.checkpoint_name}_seed{self.args.train_seed}'
                if not os.path.exists(path):
                    os.mkdir(path) 
                path = f'output_simple_env/{scene}_{self.args.checkpoint_name}_seed{self.args.train_seed}/{ep}'  
                if not os.path.exists(path):
                    os.mkdir(path)          

            while not done:
                # GET OBSERVATION
                if episode_steps == 0:
                    action = [0,0]
                obs, waypoints = self.get_observation(env, obs_encoder, mode="polar", action=action)

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
                    plt.imshow(wayypints_map, cmap='gray')
                    waypoints_in_map = list(map(env.world_to_map, env.waypoints))
                    for waypoint in waypoints_in_map:
                        plt.plot(waypoint[1], waypoint[0], marker='o')
                    plt.scatter((env.robot_pos_map[1]), (env.robot_pos_map[0]), marker=rotated_marker, facecolors='none', edgecolors='b')

                    label = f'episode = {ep} step = {episode_steps}\n' + \
                        f'robot pos: {env.robot_pos} orientation: {rotation_plot_radian} (rad) {rotation_plot_degree} (degree) \n' + \
                        f'next waypoint pos: {env.waypoints[0]}\n' + \
                        f'next waypoint distance = {waypoints[0][0]}' + \
                            f'next waypoints angle = {waypoints_angle_radian}(rad) {waypoints_angle_degree} (degree) \n'
                
                # TAKE A STEP
                if env.no_of_collisions >= 5:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs, evaluate=True)  # Sample action from policy
                reward, done, info = env.step(action) # Step

                #print("waypoint distance", waypoints[0][0])
                #print("waypoint angle", waypoints[0][1], np.degrees(waypoints[0][1]))
                #sys.exit()
                
                #print("reward", reward)
                #print("Action", action)
                #print(env.robot_pos)
                #print("waypoints", env.waypoints)
                #print("step_number", env.step_number)
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                if self.args.plot:   
                    label += f'action = {action} \n' + \
                                  f'reward = {reward} total reward = {episode_reward}'
                    plt.xlabel(label)
                    plt.savefig(f'{path}/{episode_steps}.png')
                    plt.close()
            
            
            total_num_episodes += 1
            if not info['success']:
                print("failed episode")
                if self.args.write_results:
                    failed_episode_file.write(f'{ep}\n')
            if self.args.write_results:
                result_file.write(f'ep {ep}\n success: {info["success"]}\n episode return: {episode_reward}\n steps: {episode_steps}\n')

            if info['success']:
                metrics['success_timestep'] += episode_steps
            metrics['episode_return'] += episode_reward
            for key in metrics:
                if key in info:
                    metrics[key] += info[key]
            
        
        if self.args.write_results:
            result_file.write(f'success rate: {metrics["success"]/ total_num_episodes}\n')
            result_file.write(f'Average reward: {metrics["episode_return"]/total_num_episodes}\n')
            result_file.write(f'success weighted by time steps: {metrics["success_timestep"]/metrics["success"]}\n')
            result_file.write('---------------------------------------\n')
            result_file.close()
        
                        
if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
