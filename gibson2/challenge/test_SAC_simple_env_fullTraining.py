import numpy as np
#import json
import os
import logging
import sys
import copy
import torch
import networkx as nx
import cv2
import gym
import pickle
import math
import time

from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from gibson2.utils.utils import parse_config, dist, cartesian_to_polar
from SAC.SAC import SAC
from SAC.replay_memory import ReplayMemory, PrioritizedReplayMemory
from gibson2.utils.constants import *
from encoder.obs_encoder import ObsEncoder
from simple_env import Simple_env
from simple_env_original import Simple_env_original
from occupancy_grid.occupancy_grid import ray_casting, plan, get_lidar, get_local_map_raycast, get_simple_local_map

logging.getLogger().setLevel(logging.WARNING)

class Challenge:
    def __init__(self):
        self.config_file = CONFIG_FILE
        self.training_scenes = ["Beechwood_1_int", "Ihlen_0_int", "Ihlen_1_int", "Merom_0_int"]
        self.test_scenes = ["Benevolence_0_int", "Rs_int", "Wainscott_1_int"]
        self.valdiation_scene = ["Pomaria_0_int"]

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
            local_map, pedestrian_map, path_found, visible_pedestrians = self.get_local_map(env, validation)
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
                if self.args.obs_map_lstm and first_episode == True:
                    obs_encoder.map_encoder.initialize()
                # print("task obs", task_obs)
                # print("waypoint", waypoints_obs)
                # print("map", local_map)
                # print("action", action)
                # print("pmap", pedestrian_map)
                encoded_obs = obs_encoder(task_obs, waypoints_obs, local_map, action, pedestrian_map, ped_pos_obs) 
        encoded_obs = encoded_obs.squeeze(0)
        return encoded_obs.detach().cpu().numpy(), path_found
    
    def validate(self, agent, writer, train_ep):
        #val_env = Simple_env_original(self.val_args, scene_id='Beechwood_1_int')
        metrics = {key: 0.0 for key in [
                'success', 'episode_return', 'success_timestep', 'personal_space_violation_step', 'pedestrian_collision']}
        self.val_args.train_seed = self.val_args.val_seed

        total_numsteps = 0
        total_num_episodes = 0
        for ep in range(1, self.val_args.val_episodes_per_scene+1):
            self.val_args.train_seed += 1
            val_env = Simple_env_original(self.val_args, scene_id=self.valdiation_scene[0])
            val_env.initialize_episode()
            episode_reward = 0
            episode_steps = 0
            done = False      

            while not done:
                # GET OBSERVATION
                if episode_steps == 0:
                    action = [0,0]
                    episode_start = True
                else:
                    episode_start = False
                obs, path_found = self.get_observation(val_env, self.obs_encoder, action=action, first_episode=episode_start)
                
                # TAKE A STEP
                if val_env.no_of_collisions >= 5:
                    action = val_env.action_space.sample()
                # elif self.args.replan_if_collision and path_found == False:
                #     action = [0, 0]
                else:
                    action = agent.select_action(obs, evaluate=True)  # Sample action from policy
                reward, reward_type, done, info = val_env.step(action) # Step

                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward
        
            total_num_episodes += 1
            if info['success']:
                metrics['success_timestep'] += episode_steps
            metrics['episode_return'] += episode_reward
            metrics['personal_space_violation_step'] += val_env.personal_space_violation_steps
            for key in metrics:
                if key in info:
                    metrics[key] += info[key]            
    
        writer.add_scalar('reward/val', metrics["episode_return"]/total_num_episodes, train_ep)
        writer.add_scalar('success_rate/val', metrics["success"]/ total_num_episodes, train_ep)

        # Save best validation model
        if metrics["success"]/total_num_episodes > self.validation_highest_success_rate:
            self.validation_highest_success_rate = metrics["success"]/ total_num_episodes
            agent.save_checkpoint(f'{self.args.checkpoint_name}_validation')
        if self.args.obs_train:
                self.obs_encoder.save_checkpoint(f'{self.args.checkpoint_name}_validation')

        
        print("___________________VALIDATION___________________")
        print(f'success rate: {metrics["success"]/ total_num_episodes}')
        print(f'No of successful episodes: {metrics["success"]}')
        print(f'Average reward: {metrics["episode_return"]/total_num_episodes}')
        if metrics['success'] != 0:
            print(f'Average timestep for successful episodes: {metrics["success_timestep"]/metrics["success"]}')
        if self.val_args.pedestrian_present:
            print(f'Personal space violation steps: {metrics["personal_space_violation_step"]/total_numsteps}')
            print(f'no of unsuccessful episodes: {self.val_args.val_episodes_per_scene - metrics["success"]}')
            print(f'no of episode with pedestrian collision: {metrics["pedestrian_collision"]}')
        print('_________________________________________________')

    # def normalize_reward(self, reward, min_reward, max_reward):
    #     """
    #     Normalize reward to have values between -1 and 1.
        
    #     Args:
    #         reward (float): The original reward value.
    #         min_reward (float): The minimum reward value observed.
    #         max_reward (float): The maximum reward value observed.
        
    #     Returns:
    #         float: The normalized reward value.
    #     """
    #     normalized_reward = 2.0 * (reward - min_reward) / (max_reward - min_reward) - 1.0
    #     return normalized_reward
    
    def normalize_obs(self, obs, min, max, min_range=-1):
        if min_range == 0:
            norm = (obs-min)/(max-min)
            return np.clip(norm, 0, 1)
        else:
            norm = (2*((obs-min)/(max-min))) - 1
            return np.clip(norm, -1, 1)

    def get_priority(self, reward_type):
        if reward_type == "pedestrian collision":
            return 4
        elif reward_type == "pca":
            return 3.5
        else:
            return 2.5

    def submit(self, agent, args):
        self.args = args
        self.val_args = copy.deepcopy(args)
        self.val_args.train_seed = self.val_args.val_seed
        self.val_args.num_pedestrians = self.val_args.val_no_of_pedestrians
        self.val_args.train_continue = False

        env_config = parse_config(self.config_file)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        metrics = {key: 0.0 for key in [
            'success', 'episode_return', 'personal_space_violation_step', 'pedestrian_collision']}
        
        # Initialize agent, encoder, writer and replay buffer
        low = np.array([-1,-1])
        high = np.array([1,1])
        action_space = gym.spaces.Box(low, high, dtype=np.float32)

        agent = SAC(num_inputs=256, action_space=action_space, args=args)
        if self.args.load_checkpoint == True:
            agent.load_checkpoint(ckpt_path=self.args.checkpoint_path)

        self.obs_encoder = ObsEncoder(args).to(self.args.device)

        writer = SummaryWriter(f'runs/{self.args.checkpoint_name}')

        if self.args.replay_buffer_type == "prioritized":
            memory = PrioritizedReplayMemory(args.replay_size, args.seed)
        else:
            memory = ReplayMemory(args.replay_size, args.seed)
        if self.args.load_checkpoint_memory == True:
            memory.load_buffer(save_path=self.args.checkpoint_path_memory)

        # Select map type
        if args.map == "cropped_map":
            self.get_local_map = get_simple_local_map
        elif args.map == "raycast":
            self.get_local_map = get_local_map_raycast
        elif args.map == "lidar":
            self.get_local_map = get_lidar
        
        total_numsteps = 0
        updates = 0
        total_num_episodes = 0
        previous_success_rate = 0
        no_previous_collision_episodes = 0
        scene_no = 0
        total_numsteps_scene = 0

        # if self.args.train_continue:
        #     for i in range(self.args.no_episode_trained):
        #         print(i)
        #         env.initialize_episode()
        #     env.args.train_continue = False
        
        if self.args.validation == True:
            self.validation_highest_success_rate = -np.inf

        while True:
            if total_num_episodes % self.args.scene_change_after_no_episode == 0:
                # Change scene
                args.train_seed += 1
                env = Simple_env_original(args, scene_id = self.training_scenes[scene_no % len(self.training_scenes)])
                
                if scene_no < len(self.training_scenes):
                    total_numsteps_scene = 0

                scene_no += 1
                
            #print("ep", total_num_episodes)
            env.initialize_episode()
            
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done:
                # Get observation
                if episode_steps == 0:
                    action = [0,0]
                    episode_start = True
                else:
                    episode_start = False
                obs, path_found = self.get_observation(env, self.obs_encoder, action=action, first_episode = episode_start)
                
                # Take step
                if args.start_steps > total_numsteps_scene and not self.args.train_continue:
                    action = env.action_space.sample()  # Sample random action
                elif env.no_of_collisions >= 5:
                    action = env.action_space.sample()
                # elif self.args.replan_if_collision and path_found == False:
                #     action = [0, 0]
                else:
                    if np.random.uniform() <= args.epsilon:
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

                reward, reward_type, done, info = env.step(action) # Step
                #normalizie reward
                #reward = 5*self.normalize_reward(reward, min_reward=self.args.pedestrian_collision_reward, max_reward=self.args.goal_reward)
                
                
                writer.add_scalar('reward/return_per_step', reward, total_numsteps)
                
                # Get next state observation
                next_state_obs, path_found = self.get_observation(env, self.obs_encoder, action=action)

                episode_steps += 1
                total_numsteps += 1
                total_numsteps_scene += 1
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == env_config['max_step'] else float(not done)

                # Add observations in replay buffer
                if self.args.replay_buffer_type == "prioritized":
                    priority = self.get_priority(reward_type)
                    memory.push(obs, action, reward, next_state_obs, mask, priority) # Append transition to memory
                else:
                    memory.push(obs, action, reward, next_state_obs, mask)
            
            total_num_episodes += 1
            
            # Add the metrics among episodes to get the average           
            metrics['episode_return'] += episode_reward
            if args.pedestrian_present:
                metrics['personal_space_violation_step'] += env.personal_space_violation_steps
            for key in metrics:
                if key in info:
                    metrics[key] += info[key]
            
            writer.add_scalar('reward/train', episode_reward, total_num_episodes)
            writer.add_scalar('reward/train_average_return', metrics['episode_return']/total_num_episodes, total_num_episodes)
            writer.add_scalar('success_rate/train', metrics['success']/total_num_episodes, total_num_episodes)
            
            # Print stats
            if total_num_episodes % self.args.no_ep_after_print == 0:
                agent.save_checkpoint(args.checkpoint_name)
                self.obs_encoder.save_checkpoint(args.checkpoint_name)
                memory.save_buffer(args.checkpoint_name_memory)

                print(f'scene {self.training_scenes[(scene_no-1)%len(self.training_scenes)]}')
                print(f'Total episode {total_num_episodes} total steps {total_numsteps} last episode reward {round(episode_reward, 2)}')
                for key in metrics:
                    if key == 'personal_space_violation_step':
                        avg_value = metrics[key] / total_numsteps
                    else:
                        avg_value = metrics[key] / total_num_episodes
                    print(f'Avg {key}: {round(avg_value, 5)}')
                print(f"np of successful episodes in last {self.args.no_ep_after_print} episodes: {metrics['success'] - previous_success_rate}")
                print(f"no of episodes with pedestrian collision in last {self.args.no_ep_after_print} episodes: {metrics['pedestrian_collision'] - no_previous_collision_episodes}")
                print()
                previous_success_rate = metrics['success']
                no_previous_collision_episodes = metrics["pedestrian_collision"]
            
            if self.args.validation == True and total_num_episodes % self.args.val_episode_interval == 0:
                self.validate(agent, writer, total_num_episodes)    
                        
if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
