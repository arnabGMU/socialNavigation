from pdb import run
import numpy as np
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
import time
import pickle

from scipy.ndimage import rotate
from matplotlib import pyplot as plt
from PIL import Image

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
from gibson2.utils.constants import *
from encoder.obs_encoder import ObsEncoder

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
        self.test_scenes = ['Beechwood_1_int','Merom_0_int', 'Rs_int']
    
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

    def get_simple_local_map(self, env, occupancy_grid):  
        # Cut 100x100 matrix from the occupancy grid centered around the robot         
        robot_pos = env.scene.world_to_seg_map(env.robots[0].get_position()[:-1])
        top_row = max(0, robot_pos[0]-50)
        bottom_row = min(occupancy_grid.shape[0], robot_pos[0]+50)
        left_col = max(0, robot_pos[1]-50)
        right_col = min(occupancy_grid.shape[1] , robot_pos[1]+50)
        map_cut = occupancy_grid[top_row:bottom_row, left_col:right_col]
       
       # Roate the occupancy grid by the robot's orientation
        _,_,yaw = env.robots[0].get_rpy()
        rotated_grid = rotate(map_cut, np.degrees(yaw), reshape=True, mode='nearest')
        # Rotated grid might be larger than 100x100. So make it 100x100 centered around the robot
        row_top = rotated_grid.shape[0]//2 - 50
        row_bottom = rotated_grid.shape[0]//2 + 50
        col_left = rotated_grid.shape[1]//2 - 50
        col_right = rotated_grid.shape[1]//2 + 50
        rotated_grid = rotated_grid[row_top: row_bottom, col_left:col_right]
        
        return rotated_grid
    
    def get_observation(self, env, state, scene_id, obs_encoder, mode="polar", occupancy_grid=None):
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
        
        #map_time_start = time.time()
        #map = self.get_local_map(env) # local map
        map = self.get_simple_local_map(env, occupancy_grid) # simple 100x100 cropped map from the GT map 
        #map_time_end = time.time()
        #map_time.append(map_time_end - map_time_start)

        map = torch.tensor(map)
        with torch.no_grad():
            encoded_obs = obs_encoder(task_obs.to(self.args.device).float().unsqueeze(0), \
                        waypoints.to(self.args.device).float().unsqueeze(0), \
                        map.to(self.args.device).float())
        encoded_obs = encoded_obs.squeeze(0)
        return encoded_obs.detach().cpu().numpy()
    
    def get_waypoints(self, scene_id, env, inflation_radius=2.6):
        graph_path = f'gibson2/data/gibson_challenge_data_2021/ig_dataset/scenes/{scene_id}/layout/trav_graph_inflation_radius_{inflation_radius}.pickle'
        graph = pickle.load(open(graph_path, 'rb'))

        source = env.scene.world_to_seg_map(env.robots[0].get_position()[:-1])
        target = env.scene.world_to_seg_map(env.task.target_pos[:-1])

        path = nx.astar_path(graph, tuple(source), tuple(target), heuristic=dist, weight="cost")
        point_interval = 5
        p = path[::point_interval][1:]
        
        if tuple(target) not in p:
            p.append(tuple(target))
        
        return list(map(env.scene.seg_map_to_world,map(np.array, p)))

    
    def build_traversibility_graph(self, scene_id, inflation_radius=2.6):
        path = f'gibson2/data/gibson_challenge_data_2021/ig_dataset/scenes/{scene_id}/layout/trav_graph_inflation_radius_{inflation_radius}.pickle'
        if not os.path.exists(path):
            img_path = f'gibson2/data/gibson_challenge_data_2021/ig_dataset/scenes/{scene_id}/layout/floor_trav_0_new.png'
            img = Image.open(img_path)
            occupancy_grid = np.array(img)
            occupancy_grid[occupancy_grid!=0] = 1
            inflated_grid = inflate_grid(occupancy_grid, inflation_radius, 0, 0)
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
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        task = env_config['task']
        if task == 'interactive_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'spl', 'effort_efficiency', 'ins', 'episode_return']}
        elif task == 'social_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'stl', 'psc', 'episode_return', 'success_timestep']}
        else:
            assert False, 'unknown task: {}'.format(task)

        # Make action space as a gym box
        low = np.array([-1,-1])
        high = np.array([1,1])
        action_space = gym.spaces.Box(low, high, dtype=np.float32)
        
        # Initialize agent, encoder, writer and replay buffer
        agent = SAC(num_inputs=256, action_space=action_space, args=args)
        if self.args.load_checkpoint == True:
            agent.load_checkpoint(ckpt_path=self.args.checkpoint_path, evaluate=True)
        obs_encoder = ObsEncoder(args).to(self.args.device)

        memory = ReplayMemory(args.replay_size, args.seed)
        if self.args.load_checkpoint_memory == True:
            memory.load_buffer(save_path=self.args.checkpoint_path_memory)
        
        evaluation_result_path = f'evaluation_results/{self.args.checkpoint_name}.txt'
        output_file = open(evaluation_result_path, 'w')

        num_episodes_per_scene = self.eval_episodes_per_scene
        split_dir = os.path.join(self.episode_dir, self.split)
        assert os.path.isdir(split_dir)
        num_scenes = len(os.listdir(split_dir))
        assert num_scenes > 0
        total_num_episodes = num_scenes * num_episodes_per_scene
        
        total_numsteps = 0
        total_num_episodes = 0
        for i in range(len(self.test_scenes)):
            #scene_id = self.val_scenes[epoch%len(self.val_scenes)]
            scene_id = self.test_scenes[i]
            json_file = os.path.join(split_dir, f'{scene_id}.json')
            env_config['scene_id'] = scene_id
            env_config['load_scene_episode_config'] = False
            env_config['scene_episode_config_name'] = json_file
            env = iGibsonEnv(config_file=env_config,
                             mode='headless',
                             action_timestep=1.0 / 10.0,
                             physics_timestep=1.0 / 40.0)
            
            inflation_radius = 2.6
            self.build_traversibility_graph(scene_id, inflation_radius)
            img_path = f'gibson2/data/gibson_challenge_data_2021/ig_dataset/scenes/{scene_id}/layout/floor_trav_0_new.png'
            img = Image.open(img_path)
            occupancy_grid = np.array(img)
            occupancy_grid[occupancy_grid!=0] = 1

            metrics = {key: 0.0 for key in [
                'success', 'stl', 'psc', 'episode_return', 'success_timestep']}

            if self.args.plot:
                f, axarr = plt.subplots(1,2, figsize=(50, 50))

            #output_file.write(f'{scene_id}\n')
            print(scene_id)
            for ep in range(1, 100):
                print(ep)
                #output_file.write(f'ep {ep}\n')

                if self.args.plot:
                    file_path = f'output_test_SAC/{scene_id}_{args.checkpoint_name}_{ep}'
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)

                state = env.reset()
                env.simulator.sync()

                #if ep <= 10:
                #    continue

                try:
                    self.waypoints = self.get_waypoints(scene_id, env, inflation_radius=inflation_radius)
                except:
                    print("episode skipped")
                    continue
                env.waypoints = self.waypoints
                env.num_wps_input = args.num_wps_input

                episode_reward = 0
                episode_steps = 0
                done = False

                while not done:
                    obs = self.get_observation(env, state, scene_id, obs_encoder, mode="polar", occupancy_grid=occupancy_grid.copy())
                    action = agent.select_action(obs, evaluate=True)  # Sample action from policy
                    next_state, reward, done, info = env.step(action) # Step
                    state = next_state

                    episode_steps += 1
                    total_numsteps += 1
                    episode_reward += reward

                    if self.args.plot:
                        robot_current_pos = env.robots[0].get_position()[:-1]
                        robot_current_pos_map = env.scene.world_to_seg_map(robot_current_pos)

                        waypoints = env.waypoints
                        waypoints = np.hstack((waypoints, np.zeros((len(waypoints),1))))
                        waypoints_robot_coord = [env.task.global_to_local(env, p)[:2] for p in waypoints]
                        # waypoints in polar coordinate in robot frame
                        waypoints = [np.array(cartesian_to_polar(p[0],p[1])) for p in waypoints_robot_coord]
                        wp = waypoints[0]

                        s = f'distance to the waypoint = {wp[0]}\n angle to the waypoint = {np.degrees(wp[1])} \n action = {action} \n reward {reward} total reward: {episode_reward}'

                        axarr[0].imshow(state['rgb'])
                        axarr[1].imshow(occupancy_grid, cmap='gray')
                        axarr[1].plot(robot_current_pos_map[1],robot_current_pos_map[0],marker=".", markersize=10, alpha=0.8)
                        axarr[1].set_xlabel(s, fontsize=40)

                        waypoints_map = list(map(env.scene.world_to_seg_map, env.waypoints))
                        for waypoint in waypoints_map:
                            axarr[1].plot(waypoint[1],waypoint[0],marker="x", markersize=10, alpha=0.8)
                        path = f'{file_path}/side_by_side'
                        if not os.path.exists(path):
                            os.mkdir(f'{file_path}/side_by_side')
                        f.savefig(f'{file_path}/side_by_side/{episode_steps}.jpg')
                        #plt.close('all')
                        #plt.close(plt.gcf())
                        #f.clf()
                        axarr[0].clear()
                        axarr[1].clear()
                        gc.collect()
                
                total_num_episodes += 1

                #output_file.write(f'success: {info["success"]}\n')
                #output_file.write(f'reward: {episode_reward}\n')
                #output_file.write(f'episode steps: {episode_steps}\n')
                #output_file.write(f'psc: {info["psc"]}\n\n')

                if info['success']:
                    metrics['success_timestep'] += episode_steps
                metrics['episode_return'] += episode_reward
                for key in metrics:
                    if key in info:
                        metrics[key] += info[key]
                        print("info", info[key])
                        avg_value = metrics[key] / ep
                        print('Avg {}: {}'.format(key, avg_value))
                #print(metrics)
            output_file.write(f'----------SCENE {ep} RESULT----------\n')
            output_file.write(f'success rate: {metrics["success"]/ ep}\n')
            output_file.write(f'Average reward: {metrics["episode_return"]/ep}\n')
            output_file.write(f'success weighted by time steps: {metrics["success_timestep"]/metrics["success"]}\n')
            output_file.write(f'Average psc: {metrics["psc"]/ep}\n')
            output_file.write('---------------------------------------\n\n')

            env.close()
        output_file.close()

if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
