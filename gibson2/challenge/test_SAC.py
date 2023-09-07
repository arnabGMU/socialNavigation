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

from gibson2.utils.utils import parse_config, dist, cartesian_to_polar
from gibson2.envs.igibson_env import iGibsonEnv
from SAC.SAC import SAC
from torch.utils.tensorboard import SummaryWriter
from SAC.replay_memory import ReplayMemory
from occupancy_grid.occupancy_grid import create_occupancy_grid, get_closest_free_space_to_robot, \
    a_star_search, update_map, get_robot_pos_on_grid, fill_invisible_cells_close_to_the_robot, \
    visualize_occupancy_grid, visualize_path, get_turn_angle, inflate_grid
from point_cloud.point_cloud import get_point_clouds, get_min_max_pcd
from frontier.frontier import get_frontiers, show_frontiers
from matplotlib import pyplot as plt
from gibson2.utils.constants import *
from encoder.obs_encoder import ObsEncoder
from PIL import Image
import pickle

logging.getLogger().setLevel(logging.WARNING)

#gc.set_threshold(0)
def visualize_rgb_image(rgb, show=False, store=False, output_path=None):
    plt.imshow(rgb)
    if show == True:
        plt.show()
    if store == True:
        plt.savefig(output_path)
    plt.close()

def make_dir(output_dir, scene_id, ep):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(f'{output_dir}/{scene_id}'):
        os.mkdir(f'{output_dir}/{scene_id}')
    if not os.path.exists(f'{output_dir}/{scene_id}/{ep}'):
        os.mkdir(f'{output_dir}/{scene_id}/{ep}')
        os.mkdir(f'{output_dir}/{scene_id}/{ep}/rgb')
        os.mkdir(f'{output_dir}/{scene_id}/{ep}/closest_frontier')
        os.mkdir(f'{output_dir}/{scene_id}/{ep}/frontier')
        os.mkdir(f'{output_dir}/{scene_id}/{ep}/global_map')
        os.mkdir(f'{output_dir}/{scene_id}/{ep}/og')
        os.mkdir(f'{output_dir}/{scene_id}/{ep}/path')
        os.mkdir(f'{output_dir}/{scene_id}/{ep}/path_unknown')
        os.mkdir(f'{output_dir}/{scene_id}/{ep}/inf_grid')

class Challenge:
    def __init__(self):
        self.config_file = CONFIG_FILE
        self.split = SPLIT
        self.episode_dir = EPISODE_DIR
        self.eval_episodes_per_scene = 10000

        self.train_scenes = ['Beechwood_1_int', 'Benevolence_0_int', 'Ihlen_1_int', 'Wainscott_1_int']
        self.train_scenes = ['Beechwood_1_int']
        self.val_scenes = ['Ihlen_0_int']
        self.test_scenes = ['Merom_0_int', 'Rs_int']
    
    def get_local_map(self, env):
        # Get point cloud in robot coordinate frame
        pc = get_point_clouds(env, visualize=False, mode="robot_coordinate")
        pc = np.vstack((pc, np.array([0,0,0]))) #Add robot position
        min_x,max_x,min_y,max_y,min_z,max_z = get_min_max_pcd(pc)

        robot_position_wc = env.robots[0].get_position()
        occupancy_grid, robot_pos, goal_pos = create_occupancy_grid([pc],min_x,max_x,min_y,max_y,min_z,max_z, \
            robot_position_wc, goal_pos=env.task.target_pos, RESOLUTION = RESOLUTION, visualize=False, index=None,\
                  output_dir=None, mode="robot_coordinate")
        #occupancy_grid = inflate_grid(occupancy_grid, 2, 0, 0)
        #occupancy_grid = fill_invisible_cells_close_to_the_robot(occupancy_grid, robot_pos[1], robot_pos[0])
        
        # Add robot footprint
        robot_footprint_radius = 0.32
        robot_footprint_radius_map = int(robot_footprint_radius/RESOLUTION)
        cv2.circle(img=occupancy_grid, center=robot_pos[::-1], radius=robot_footprint_radius_map,color=1, thickness=-1)

        # Align occupancy grid in a larger matrix of size (224,224)
        mask_grid_size = (224,224)
        occupancy_grid_mask = np.ones(mask_grid_size)*0.5
        i_x = int(mask_grid_size[0]/2)-robot_pos[0]
        i_y = int(mask_grid_size[1]/2)-robot_pos[1]
        try:
            occupancy_grid_mask[i_x:i_x+occupancy_grid.shape[0],i_y:i_y+occupancy_grid.shape[1]] = occupancy_grid
        except:
            print(occupancy_grid.shape)
            plt.imshow(occupancy_grid); plt.show()
            print(robot_pos)
        occupancy_grid = occupancy_grid_mask
        
        return occupancy_grid

    def get_observation(self, env, state, scene_id, obs_encoder, mode="polar"):
        #task_obs = torch.tensor(state['task_obs']) # relative goal position, orientation, linear and angular velocity
        if mode == "cartesian":
            task_obs = torch.tensor(env.task.target_pos[:-1] - env.robots[0].get_position()[:-1])
        else:
            task_obs = torch.tensor(state['task_obs'][:4]) # relative goal position, orientation, linear and angular velocity

        # Get waypoints
        waypoints = env.waypoints[:self.args.num_wps_input]
        while len(waypoints) < self.args.num_wps_input:
            waypoints.append(waypoints[-1])
        if mode == "cartesian":
            waypoints -= env.robots[0].get_position()[:-1]
        else:
            waypoints = np.hstack((waypoints, np.zeros((len(waypoints),1))))
            waypoints_robot_coord = [env.task.global_to_local(env, p)[:2] for p in waypoints]
            # waypoints in polar coordinate in robot frame
            waypoints = [np.array(cartesian_to_polar(p[0],p[1])) for p in waypoints_robot_coord] 

        waypoints = torch.tensor(np.array(waypoints))
        waypoints = waypoints.reshape((-1,))
        
        map = self.get_local_map(env)
        map = torch.tensor(map)
        with torch.no_grad():
            encoded_obs = obs_encoder(task_obs.to(self.args.device).float().unsqueeze(0), \
                        waypoints.to(self.args.device).float().unsqueeze(0), \
                        map.to(self.args.device).float())
        encoded_obs = encoded_obs.squeeze(0)
        return encoded_obs.detach().cpu().numpy()
    
    def get_waypoints(self, scene_id, env):
        graph_path = f'gibson2/data/gibson_challenge_data_2021/ig_dataset/scenes/{scene_id}/layout/trav_graph.pickle'
        graph = pickle.load(open(graph_path, 'rb'))

        source = env.scene.world_to_seg_map(env.robots[0].get_position()[:-1])
        target = env.scene.world_to_seg_map(env.task.target_pos[:-1])

        path = nx.astar_path(graph, tuple(source), tuple(target), heuristic=dist, weight="cost")
        point_interval = 5
        p = path[::point_interval][1:]
        
        if tuple(target) not in p:
            p.append(tuple(target))
        
        return list(map(env.scene.seg_map_to_world,map(np.array, p)))

    
    def build_traversibility_graph(self, scene_id):
        path = f'gibson2/data/gibson_challenge_data_2021/ig_dataset/scenes/{scene_id}/layout/trav_graph.pickle'
        if not os.path.exists(path):
            img_path = f'gibson2/data/gibson_challenge_data_2021/ig_dataset/scenes/{scene_id}/layout/floor_trav_0_new.png'
            img = Image.open(img_path)
            occupancy_grid = np.array(img)
            occupancy_grid[occupancy_grid!=0] = 1
            inflated_grid = inflate_grid(occupancy_grid, 3, 0, 0)
            graph = nx.grid_graph((inflated_grid.shape[0], inflated_grid.shape[1]))
            nx.set_edge_attributes(graph, np.inf, "cost")

            free_space = np.vstack((np.where(inflated_grid==1)[0], np.where(inflated_grid==1)[1])).T
            free_space_nodes = tuple(map(tuple, free_space))
            free_space_nodes = {node:None for node in free_space_nodes}

            cost = {}
            for edge in graph.edges():
                if edge[0] in free_space_nodes and edge[1] in free_space_nodes:
                    cost[edge] = np.linalg.norm(np.array(edge[0])-np.array(edge[1]))
            nx.set_edge_attributes(graph, cost, "cost")
            
            pickle.dump(graph, open(path, 'wb'))
        else:
            graph = pickle.load(open(path, 'rb'))
        
        return graph


    def submit(self, agent, args):
        self.args = args
        env_config = parse_config(self.config_file)
        val_env_config = parse_config(self.config_file)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        task = env_config['task']
        if task == 'interactive_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'spl', 'effort_efficiency', 'ins', 'episode_return']}
        elif task == 'social_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'stl', 'psc', 'episode_return']}
        else:
            assert False, 'unknown task: {}'.format(task)

        # Make action space as a gym box
        low = np.array([-1,-1])
        high = np.array([1,1])
        action_space = gym.spaces.Box(low, high, dtype=np.float32)
        
        # Initialize agent, encoder, writer and replay buffer
        agent = SAC(num_inputs=256, action_space=action_space, args=args)
        if self.args.load_checkpoint == True:
            agent.load_checkpoint(ckpt_path=self.args.checkpoint_path)
        obs_encoder = ObsEncoder(args).to(self.args.device)
        writer = SummaryWriter('runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                    args.policy, "autotune" if args.automatic_entropy_tuning else ""))
        memory = ReplayMemory(args.replay_size, args.seed)
        if self.args.load_checkpoint_memory == True:
            memory.load_buffer(save_path=self.args.checkpoint_path_memory)

        num_episodes_per_scene = self.eval_episodes_per_scene
        split_dir = os.path.join(self.episode_dir, self.split)
        assert os.path.isdir(split_dir)
        num_scenes = len(os.listdir(split_dir))
        assert num_scenes > 0
        total_num_episodes = num_scenes * num_episodes_per_scene

        # Intialize eval scene
        if args.eval == True:
            val_scene_id = self.val_scenes[0]
            json_file = os.path.join(split_dir, f'{val_scene_id}.json')
            val_env_config['scene_id'] = val_scene_id
            val_env_config['load_scene_episode_config'] = False
            val_env_config['scene_episode_config_name'] = json_file
            val_env = iGibsonEnv(config_file=val_env_config,
                                mode='headless',
                                action_timestep=1.0 / 10.0,
                                physics_timestep=1.0 / 40.0)
            
            self.build_traversibility_graph(val_scene_id)
        
        total_numsteps = 0
        updates = 0
        epoch = 0
        total_num_episodes = 0
        highest_reward = -np.inf
        end = False
        while True:
            scene_id = self.train_scenes[epoch%len(self.train_scenes)]
            json_file = os.path.join(split_dir, f'{scene_id}.json')
            env_config['scene_id'] = scene_id
            env_config['load_scene_episode_config'] = False
            env_config['scene_episode_config_name'] = json_file
            env = iGibsonEnv(config_file=env_config,
                             mode='headless',
                             action_timestep=1.0 / 10.0,
                             physics_timestep=1.0 / 40.0)
            
            self.build_traversibility_graph(scene_id)
            
            for ep in range(1, num_episodes_per_scene+1):
                #print(f'{scene_id} epoch: {epoch} Episode: {ep}/{num_episodes_per_scene} num_steps: {total_numsteps}')
                
                state = env.reset()
                env.simulator.sync()
                try:
                    self.waypoints = self.get_waypoints(scene_id, env)
                except Exception as e:
                    print("episode skipped", e)
                    continue
                env.waypoints = self.waypoints
                env.num_wps_input = args.num_wps_input

                episode_reward = 0
                episode_steps = 0
                done = False

                while not done:
                    obs = self.get_observation(env, state, scene_id, obs_encoder, mode="polar")
                    if args.start_steps > total_numsteps:
                        action = env.action_space.sample()  # Sample random action
                    else:
                        action = agent.select_action(obs)  # Sample action from policy

                    if len(memory) > args.batch_size:
                        # Number of updates per step in environment
                        for i in range(args.updates_per_step):
                            # Update parameters of all the networks
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                            writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                            writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                            writer.add_scalar('loss/policy', policy_loss, updates)
                            writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                            writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                            updates += 1

                    next_state, reward, done, info = env.step(action) # Step
                    next_state_obs = self.get_observation(env, next_state, scene_id, obs_encoder, mode="polar")
                    #next_state_obs = next_state_obs.squeeze(0)

                    episode_steps += 1
                    total_numsteps += 1
                    episode_reward += reward

                    # Ignore the "done" signal if it comes from hitting the time horizon.
                    # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                    mask = 1 if episode_steps == env_config['max_step'] else float(not done)

                    memory.push(obs, action, reward, next_state_obs, mask) # Append transition to memory

                    state = next_state
                
                total_num_episodes += 1
                
                metrics['episode_return'] += episode_reward
                for key in metrics:
                    if key in info:
                        metrics[key] += info[key]
                
                writer.add_scalar('reward/train', episode_reward, total_num_episodes)
                #agent.save_checkpoint("current_polar_reversedone")
                #memory.save_buffer("current_polar_reversedone_memory")
                #print("Scene {} Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(total_num_episodes, total_numsteps, episode_steps, round(episode_reward, 2)))
                if ep % 30 == 0:
                    agent.save_checkpoint(args.checkpoint_name)
                    memory.save_buffer(args.checkpoint_name_memory)
                    print(f'Scene {scene_id} epoch {epoch} Episode: {ep}/{num_episodes_per_scene} Total episode {total_num_episodes} total steps {total_numsteps} reward {round(episode_reward, 2)}\n')
                    for key in metrics:
                        avg_value = metrics[key] / total_num_episodes
                        print('Avg {}: {}'.format(key, avg_value))
                    print()

                if total_numsteps > args.num_steps:
                    end = True
                    break

                #-------------------------------------------Eval------------------------------------------#

                if total_num_episodes % 50 == 0 and args.eval is True:
                    metrics_val = {key: 0.0 for key in [
                        'success', 'stl', 'psc', 'episode_return']}
                    avg_reward = 0.
                    episodes = 5
                    for _  in range(episodes):
                        state = val_env.reset()
                        waypoints = self.get_waypoints(val_scene_id, val_env)
                        val_env.waypoints = waypoints
                        val_env.num_wps_input = args.num_wps_input

                        episode_reward = 0
                        done = False
                        while not done:
                            obs = self.get_observation(val_env, state, val_scene_id, obs_encoder, mode="polar")
                            action = agent.select_action(obs, evaluate=True)

                            next_state, reward, done, info = val_env.step(action)
                            episode_reward += reward

                            state = next_state
                        avg_reward += episode_reward
                        metrics_val['episode_return'] += episode_reward
                        for key in metrics_val:
                            if key in info:
                                metrics_val[key] += info[key]
                    avg_reward /= episodes

                    writer.add_scalar('avg_reward/test', avg_reward, total_num_episodes)
                    if avg_reward > highest_reward:
                        highest_reward = avg_reward
                        agent.save_checkpoint("highest_val")
                    
                    print("----------------------------------------")
                    print(f'Scene {val_scene_id} epoch {epoch} Episode: {ep}/{num_episodes_per_scene} \
                      Total episode {total_num_episodes} total steps {total_numsteps}')
                    print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
                    for key in metrics_val:
                        metrics_val[key] /= episodes
                        print('Avg {}: {}'.format(key, metrics_val[key]))
                    print("----------------------------------------")
            epoch += 1
            env.close()
            if end == True:
                break
        
        for key in metrics:
            metrics[key] /= total_num_episodes
            print('Avg {}: {}'.format(key, metrics[key]))
                        
if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
