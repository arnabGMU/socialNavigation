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

from gibson2.utils.utils import parse_config
from gibson2.envs.igibson_env import iGibsonEnv
logging.getLogger().setLevel(logging.WARNING)


from occupancy_grid.occupancy_grid import create_occupancy_grid, get_closest_free_space_to_robot, \
    a_star_search, update_map, get_robot_pos_on_grid, fill_invisible_cells_close_to_the_robot, \
    visualize_occupancy_grid, visualize_path, get_turn_angle, inflate_grid
from point_cloud.point_cloud import get_point_clouds, get_min_max_pcd
from frontier.frontier import get_frontiers, show_frontiers
from matplotlib import pyplot as plt

CONFIG_FILE = './gibson2/examples/configs/locobot_social_nav.yaml'
SPLIT = 'train'
EPISODE_DIR = 'gibson2/data/gibson_challenge_data_2021/episodes_data/social_nav'
TASK= 'social'
OUTPUT_DIR = 'output3'

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
        self.eval_episodes_per_scene = 100

    def submit(self, agent):
        env_config = parse_config(self.config_file)

        task = env_config['task']
        if task == 'interactive_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'spl', 'effort_efficiency', 'ins', 'episode_return']}
        elif task == 'social_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'stl', 'psc', 'episode_return']}
        else:
            assert False, 'unknown task: {}'.format(task)

        num_episodes_per_scene = self.eval_episodes_per_scene
        split_dir = os.path.join(self.episode_dir, self.split)
        assert os.path.isdir(split_dir)
        num_scenes = len(os.listdir(split_dir))
        assert num_scenes > 0
        total_num_episodes = num_scenes * num_episodes_per_scene

        idx = 0
        
        for json_file in os.listdir(split_dir)[0:]:
            scene_id = json_file.split('.')[0]
            json_file = os.path.join(split_dir, json_file)

            env_config['scene_id'] = scene_id
            env_config['load_scene_episode_config'] = True
            env_config['scene_episode_config_name'] = json_file
            env = iGibsonEnv(config_file=env_config,
                             mode='headless',
                             action_timestep=1.0 / 10.0,
                             physics_timestep=1.0 / 40.0)
            for ep in range(num_episodes_per_scene):
                idx += 1
                print('Episode: {}/{}'.format(idx, total_num_episodes))
                try:
                    agent.reset()
                except:
                    pass
                state = env.reset()
                #env.robots[0].set_position([-2,0,0])
                #env.robots[0].set_orientation([0,0,0,1])
                env.simulator.sync()
                state = env.get_state()
                
                episode_return = 0.0
                i = 0
                angles = []
                occupancy_grid_prev = None
                robot_pos_prev_map = None
                robot_position_wc_prev = None
                step_threshold = 0
                goal_found = False
                
                make_dir(OUTPUT_DIR, scene_id, ep)
                out_dir = f'{OUTPUT_DIR}/{scene_id}/{ep}'
                while True:
                    print(f'i = {i}')
                    print(f"step threshold {step_threshold}")
                    visualize_rgb_image(state['rgb'], show=False, store=True, output_path=f'{out_dir}/rgb/{i}')
                    
                    # Get point clouds
                    pc = get_point_clouds(env, visualize=False, mode="world_coordinate")
                    if i == 0:
                        goal_wc = env.task.target_pos
                        pc = np.vstack((pc, goal_wc))
                        pc = np.vstack((pc, env.robots[0].get_position()))
                        print("goal", goal_wc)
                        min_x,max_x,min_y,max_y,min_z,max_z = get_min_max_pcd(pc)
                    else:
                        min_x,max_x,min_y,max_y,min_z,max_z = get_min_max_pcd(pc,min_x,max_x,min_y,max_y,min_z,max_z)
                    
                    # Get partial map from current observation and inflate grid
                    robot_position_wc = env.robots[0].get_position()
                    occupancy_grid, robot_pos, goal_pos = \
                        create_occupancy_grid([pc],min_x,max_x,min_y,max_y,min_z,max_z, robot_position_wc, goal_wc, \
                                              RESOLUTION = 0.05, visualize=True, index=i, output_dir=out_dir, mode="world_coordinate")
                    occupancy_grid = inflate_grid(occupancy_grid, 2, 0, 0)
                    visualize_occupancy_grid(occupancy_grid, robot_pos[::-1], goal_pos[::-1], store=True, output_path=f'{out_dir}/inf_grid/{i}')
                
                    # Make a global map observed so far using the current partial map and global map
                    # Fill invisible cells close to the robot       
                    prev_robot_pos_current_map = get_robot_pos_on_grid(robot_position_wc_prev, min_x, min_y, RESOLUTION = 0.05)
                    occupancy_grid = update_map(occupancy_grid, occupancy_grid_prev, robot_pos_prev_map, prev_robot_pos_current_map)
                    occupancy_grid = fill_invisible_cells_close_to_the_robot(occupancy_grid, robot_pos[1], robot_pos[0])
                    visualize_occupancy_grid(occupancy_grid, robot_pos[::-1], goal_pos[::-1], store=True, output_path=f'{out_dir}/global_map/{i}')
                    
                    # Store the current occupancy grid, robot positions on the map and wc for updating the map in the next iteraiton
                    occupancy_grid_prev = occupancy_grid
                    robot_pos_prev_map = robot_pos
                    robot_position_wc_prev = robot_position_wc

                    # Get closest free space to robot where we need to navigate to the frontier or goal
                    closest_free_space = get_closest_free_space_to_robot(occupancy_grid, robot_pos)
                    frontiers = get_frontiers(occupancy_grid, closest_free_space=closest_free_space, closest= True, frontier_size=15)
                    show_frontiers(occupancy_grid, frontiers, show=False, store=True, output_path=f'{out_dir}/frontier/{i}')
                    if goal_pos in tuple(map(tuple,closest_free_space)):
                        goal_found = True

                    # Do not need to replan when the robot is turning to save computation time on A* planning
                    if not angles:
                        # Plan after every 10 steps 
                        if step_threshold == 0:   
                            a_star_path, min_cost = a_star_search(occupancy_grid, frontiers, robot_pos, goal_pos, i, goal_found=goal_found, output_dir=out_dir)
                            print(f'robot position {a_star_path[0]} next point {a_star_path[1]}')
                            #visualize_path(occupancy_grid, a_star_path, show=False, store=True, output_path=f'{out_dir}/path/{i}')
                            current_point = a_star_path.pop(0)
                            next_point = a_star_path[0]
                            #x = current_point[1][1] - next_point[0][1]
                            #y = next_point[0][0] - current_point[1][0]
                            x = next_point[1] - current_point[1]
                            y = current_point[0] - next_point[0]
                            step_threshold += 1
                            
                        else:
                            visualize_path(occupancy_grid, a_star_path, show=False, store=True, output_path=f'{out_dir}/path/{i}')
                            print(f'robot position {a_star_path[0]} next point {a_star_path[1]}')                        
                            current_point = a_star_path.pop(0)
                            next_point = a_star_path[0]
                            x = next_point[1] - current_point[1]
                            y = current_point[0] - next_point[0]
                            step_threshold += 1
                            if step_threshold == 10:
                                step_threshold = 0
                    
                    
                                 
                    # Calculate the required turn angle
                    angle = np.arctan2(y,x)
                    print("angle", angle)
                    _,_,yaw = env.robots[0].get_rpy()
                    print("yaw", yaw)
                    turn_angle = get_turn_angle(angle, yaw)
                    print("turn angle", turn_angle)
                    '''
                    if angle > 3 and yaw > 1:
                        turn_angle = angle - yaw
                    else:
                        turn_angle = angle + yaw
                    if angle < -1 and yaw < -3:
                        turn_angle = angle - yaw
                    if angle < -3 and yaw < -1:
                        turn_angle = angle 
                    '''
                    if min_cost == np.inf:
                        print("no path found")
                        step_threshold = 0
                    else:
                        if abs(turn_angle) > 0.5 or angles: # robot needs to start turning or still turning
                            step_threshold = 0
                            if not angles:
                                angles = [turn_angle/10] * 10
                                position_before_turning = robot_position_wc
                                print(f"position before turning {position_before_turning}")
                            print(f'turn no {len(angles)}: angle {angles[0]}')
                            print(f'robot position {a_star_path[0]} next point {a_star_path[1]}')
                            env.robots[0].turn_left(angles.pop(0))
                            if len(angles) == 0:
                                print(f'after turn robot position {env.robots[0].get_position()}')
                                env.robots[0].set_position(position_before_turning)
                                env.simulator.sync()
                                print(f'after turn fixing robot position {env.robots[0].get_position()}')

                            #env.robots[0].move_forward()    
                        else:
                            env.robots[0].turn_left(turn_angle)
                            env.robots[0].move_forward()

                    #action = env.action_space.sample()
                    #action = agent.act(state)
                    state, reward, done, info = env.step(np.array([0,0]))
                    episode_return += reward
                    if done:
                        print(reward)
                        print(info)
                        break
                    i  += 1

                    if i==10:
                        break

                    print(psutil.cpu_percent())
                    print(psutil.virtual_memory().percent)
                    
                    del pc
                    del occupancy_grid
                    del closest_free_space
                    for f in frontiers:
                        del f
                    del frontiers
                    del turn_angle
                    n = gc.collect()
                    print("Number of unreachable objects collected by GC:", n)
                    print("Uncollectable garbage:", gc.garbage)
                    print()

                metrics['episode_return'] += episode_return
                for key in metrics:
                    if key in info:
                        metrics[key] += info[key]

        for key in metrics:
            metrics[key] /= total_num_episodes
            print('Avg {}: {}'.format(key, metrics[key]))


if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
