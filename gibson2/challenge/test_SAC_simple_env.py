import numpy as np
#import json
import os
import logging
import sys
import copy
import open3d as o3d
import itertools
import torch
import gc
import psutil
import datetime
import networkx as nx
import cv2
import gym
import pickle

from scipy.ndimage import rotate
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from PIL import Image

from gibson2.utils.utils import parse_config, dist, cartesian_to_polar
from SAC.SAC import SAC
from SAC.replay_memory import ReplayMemory
from gibson2.utils.constants import *
from encoder.obs_encoder import ObsEncoder
from simple_env import Simple_env
from simple_env_original import Simple_env_original

logging.getLogger().setLevel(logging.WARNING)

class Challenge:
    def __init__(self):
        self.config_file = CONFIG_FILE
        self.split = SPLIT
        self.episode_dir = EPISODE_DIR
        self.eval_episodes_per_scene = 1000
    
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
            waypoints = [np.array([env.get_relative_pos(p), env.get_relative_orientation(p)]) for p in waypoints] 

        waypoints = torch.tensor(np.array(waypoints))
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
        return encoded_obs.detach().cpu().numpy()

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
                'success', 'episode_return']}
        else:
            assert False, 'unknown task: {}'.format(task)
        
        # Initialize agent, encoder, writer and replay buffer
        low = np.array([-1,-1])
        high = np.array([1,1])
        action_space = gym.spaces.Box(low, high, dtype=np.float32)
        agent = SAC(num_inputs=256, action_space=action_space, args=args)
        if self.args.load_checkpoint == True:
            agent.load_checkpoint(ckpt_path=self.args.checkpoint_path)
        obs_encoder = ObsEncoder(args).to(self.args.device)
        writer = SummaryWriter(f'runs/{self.args.checkpoint_name}')
        memory = ReplayMemory(args.replay_size, args.seed)
        if self.args.load_checkpoint_memory == True:
            memory.load_buffer(save_path=self.args.checkpoint_path_memory)
        
        total_numsteps = 0
        updates = 0
        total_num_episodes = 0
        env = Simple_env_original(args, scene_id = 'Beechwood_1_int')
        #env = Simple_env(args, scene_id = 'Beechwood_1_int')

        if self.args.train_continue:
            for i in range(self.args.no_episode_trained):
                print(i)
                env.initialize_episode()
            env.args.train_continue = False

        while True:
            #print(f'{scene_id} epoch: {epoch} Episode: {ep}/{num_episodes_per_scene} num_steps: {total_numsteps}')
            env.initialize_episode()

            episode_reward = 0
            episode_steps = 0
            done = False

            while not done:
                if episode_steps == 0:
                    action = [0,0]
                obs = self.get_observation(env, obs_encoder, mode="polar", action=action)
                if args.start_steps > total_numsteps and not self.args.train_continue:
                    action = env.action_space.sample()  # Sample random action
                elif env.no_of_collisions >= 5:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs)  # Sample action from policy

                if len(memory) > args.batch_size:
                    # Number of updates per step in environment
                    for i in range(args.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                        writer.add_scalar('loss/critic_1', critic_1_loss, updates) #changed
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                        updates += 1

                reward, done, info = env.step(action) # Step
                #print("reward", reward)
                #print("waypoints", env.waypoints)
                #print("step_number", env.step_number)
                next_state_obs = self.get_observation(env, obs_encoder, mode="polar", action=action)

                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == env_config['max_step'] else float(not done)

                memory.push(obs, action, reward, next_state_obs, mask) # Append transition to memory
            
            total_num_episodes += 1
            
            metrics['episode_return'] += episode_reward
            for key in metrics:
                if key in info:
                    metrics[key] += info[key]
            
            writer.add_scalar('reward/train', episode_reward, total_num_episodes)
            #agent.save_checkpoint("current_polar_reversedone")
            #memory.save_buffer("current_polar_reversedone_memory")
            #print("Scene {} Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(total_num_episodes, total_numsteps, episode_steps, round(episode_reward, 2)))
            if total_num_episodes % 30 == 0:
                agent.save_checkpoint(args.checkpoint_name)
                memory.save_buffer(args.checkpoint_name_memory)
                print(f'Total episode {total_num_episodes} total steps {total_numsteps} last episode reward {round(episode_reward, 2)}')
                for key in metrics:
                    avg_value = metrics[key] / total_num_episodes
                    print('Avg {}: {}'.format(key, avg_value))
                print()
    
                        
if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
