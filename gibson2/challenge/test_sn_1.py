from gibson2.utils.utils import parse_config
import numpy as np
#import json
import os
from gibson2.envs.igibson_env import iGibsonEnv
import logging
import sys
import open3d as o3d
logging.getLogger().setLevel(logging.WARNING)
import gc
import psutil

from occupancy_grid.occupancy_grid import create_occupancy_grid, get_closest_free_space_to_robot, a_star_search
from point_cloud.point_cloud import get_point_clouds, get_min_max_pcd
from frontier.frontier import get_frontiers, show_frontiers
from matplotlib import pyplot as plt

CONFIG_FILE = './gibson2/examples/configs/locobot_social_nav.yaml'
SPLIT = 'train'
EPISODE_DIR = 'gibson2/data/gibson_challenge_data_2021/episodes_data/social_nav'
TASK= 'social'


def visualize_rgb_image(rgb, show=False, store=False, output_path=None):
    plt.imshow(rgb)
    if show == True:
        plt.show()
    if store == True:
        plt.savefig(output_path)
    plt.close()

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
        angles = []
        for json_file in os.listdir(split_dir):
            scene_id = json_file.split('.')[0]
            json_file = os.path.join(split_dir, json_file)

            env_config['scene_id'] = scene_id
            env_config['load_scene_episode_config'] = True
            env_config['scene_episode_config_name'] = json_file
            env = iGibsonEnv(config_file=env_config,
                             mode='headless',
                             action_timestep=1.0 / 10.0,
                             physics_timestep=1.0 / 40.0)
            for _ in range(num_episodes_per_scene):
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
                while True:
                    print(f'i = {i}')
                    #visualize_rgb_image(state['rgb'], show=False, store=True, output_path=f'output/rgb/{i}')
                    plt.imshow(state['rgb'])
                    plt.savefig(f'output/rgb/{i}')
                    plt.close()
                    
                    pc = get_point_clouds(env, i, visualize=False)
                    if i == 0:
                        goal = env.task.target_pos
                        pcd = np.vstack((pc, goal))
                        print("goal", goal)
                    else:
                        pcd = np.unique(np.vstack((pcd, pc)),axis=0)
                        #pcd = np.vstack((pc, goal))
                    #np.save(f'pc/pc_{i}', pcd)
                    print(pcd.shape)
                    
                    min_x,max_x,min_y,max_y,min_z,max_z = get_min_max_pcd([pcd])
                    #print(f"min_z {min_z} max_z {max_z}")
                    
                    robot_position = env.robots[0].get_position()
                    occupancy_grid, robot_pos, goal_pos = \
                        create_occupancy_grid([pcd],min_x,max_x,min_y,max_y,min_z,max_z, robot_position, goal, RESOLUTION = 0.05, visualize=True, index=i)
                    
                    if not angles:
                        closest_free_space = get_closest_free_space_to_robot(occupancy_grid, robot_pos)
                        
                        frontiers = get_frontiers(occupancy_grid, closest_free_space=closest_free_space, closest= True, frontier_size=15)
                        frontier_path = f'output/frontier/{i}'
                        show_frontiers(occupancy_grid, frontiers, show=False, store=True, output_path=frontier_path)

                        a_star_path = a_star_search(occupancy_grid, frontiers, robot_pos, goal_pos, i)
                        x = a_star_path[1][1] - a_star_path[0][1]
                        y = a_star_path[1][0] - a_star_path[0][0]
                        
                    print(f'robot position {a_star_path[0]} next point {a_star_path[1]}')
                    angle = np.arctan2(y,x)
                    print("angle", angle)
                    #env.robots[0].set_orientation([0,0,0,1])
                    _,_,yaw = env.robots[0].get_rpy()
                    print("yaw", yaw)

                    if angle > 3 and yaw > 1:
                        turn_angle = angle - yaw
                    else:
                        turn_angle = angle + yaw
                    if abs(turn_angle) > 1 or angles: # robot needs to start turning or still turning
                        if not angles:
                            #if angle > 3:
                            #    angle *= -1
                            #if yaw > 3:
                            #    yaw *= -1
                            angles = [turn_angle/10] * 10
                        print(f'turn no {len(angles)}: angle {angles[0]}')
                        env.robots[0].turn_left(angles.pop(0))
                        #env.robots[0].move_forward()
                    
                    else:
                        #env.robots[0].turn_left(turn_angle)
                        env.robots[0].move_forward()

                    #action = env.action_space.sample()
                    #action = agent.act(state)
                    state, reward, done, info = env.step(np.array([0,0]))
                    episode_return += reward
                    if done:
                        break
                    i  += 1
                    print(psutil.cpu_percent())
                    print(psutil.virtual_memory().percent)
                    print()
                    gc.collect()

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
