import numpy as np
import os
import logging
import torch
import datetime
import cv2
import gym

from gibson2.utils.utils import parse_config
from SAC.SAC import SAC
from SAC.replay_memory import ReplayMemory
from occupancy_grid.occupancy_grid import visualize_path
from matplotlib import pyplot as plt
from gibson2.utils.constants import *
from encoder.obs_encoder import ObsEncoder
from PIL import Image
from simple_env import Simple_env


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
        self.eval_episodes_per_scene = 1000

    def get_observation(self, env, obs_encoder, mode="polar"):
        #task_obs = torch.tensor(state['task_obs']) # relative goal position, orientation, linear and angular velocity
        if mode == "cartesian":
            task_obs = torch.tensor(env.task.target_pos[:-1] - env.robots[0].get_position()[:-1])
        else:
            relative_goal_pos = env.get_relative_pos(env.goal_pos)
            relative_goal_orientation = env.get_relative_orientation(env.goal_pos)
            task_obs = torch.tensor([relative_goal_pos, relative_goal_orientation]) # relative goal position, orientation, linear and angular velocity

        # Get waypoints
        waypoints = env.waypoints[:self.args.num_wps_input]
        if len(waypoints) == 0:
            waypoints = [env.goal_pos]
        while len(waypoints) < self.args.num_wps_input:
            waypoints.append(waypoints[-1])
        if mode == "cartesian":
            waypoints -= env.robots[0].get_position()[:-1]
        else:
            # waypoints in polar coordinate in robot frame
            waypoints = [np.array([env.get_relative_pos(p), env.get_relative_orientation(p)]) for p in waypoints] 

        waypoints = torch.tensor(np.array(waypoints))
        waypoints = waypoints.reshape((-1,))
        
        with torch.no_grad():
            encoded_obs = obs_encoder(task_obs.to(self.args.device).float().unsqueeze(0), \
                        waypoints.to(self.args.device).float().unsqueeze(0))
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
            agent.load_checkpoint(ckpt_path=self.args.checkpoint_path, evaluate=True)
        obs_encoder = ObsEncoder(args).to(self.args.device)
        memory = ReplayMemory(args.replay_size, args.seed)
        if self.args.load_checkpoint_memory == True:
            memory.load_buffer(save_path=self.args.checkpoint_path_memory)

        num_episodes_per_scene = self.eval_episodes_per_scene
        split_dir = os.path.join(self.episode_dir, self.split)
        assert os.path.isdir(split_dir)
        num_scenes = len(os.listdir(split_dir))
        assert num_scenes > 0
        total_num_episodes = num_scenes * num_episodes_per_scene
        
        total_numsteps = 0
        updates = 0
        total_num_episodes = 0
        env = Simple_env(args)
        wayypints_map = np.ones((int((env.x_range[1]-env.x_range[0])/env.resolution), int((env.y_range[1]-env.y_range[0])/env.resolution)))
        for ep in range(1,11):
            env.initialize_episode()
            episode_reward = 0
            episode_steps = 0
            done = False
            path = f'output_simple_env_seed1234/{ep}'
            if ep == 1: continue
            if not os.path.exists(path):
                os.mkdir(path)

            while not done:
                obs = self.get_observation(env, obs_encoder, mode="polar")
                action = agent.select_action(obs, evaluate=True)  # Sample action from policy
                reward, done, info = env.step(action) # Step
                
                #print("reward", reward)
                #print("Action", action)
                #print(env.robot_pos)
                #print("waypoints", env.waypoints)
                #print("step_number", env.step_number)
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                plt.figure(figsize=(12,12))
                plt.imshow(wayypints_map)
                plt.plot(env.robot_pos_map[1], env.robot_pos_map[0], marker='*')
                waypoints_in_map = list(map(env.world_to_map, env.waypoints))
                for waypoint in waypoints_in_map:
                    plt.plot(waypoint[1], waypoint[0], marker='o')
                label = f'episode = {ep} step = {episode_steps}\n reward = {reward} total reward = {episode_reward}'
                plt.xlabel(label)
                plt.savefig(f'{path}/{episode_steps}.png')
                plt.close()
            
            total_num_episodes += 1
            
            #metrics['episode_return'] += episode_reward
            #for key in metrics:
            #    if key in info:
            #        metrics[key] += info[key]
    
                        
if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
