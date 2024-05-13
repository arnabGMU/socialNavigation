import argparse
import sys
#sys.path.appeng('../iGibsonChallenge2021/')

from gibson2.challenge.test_SAC_simple_env_fullTraining import Challenge

def main():
    parser = argparse.ArgumentParser()
    
    # SAC
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)') # changed
    parser.add_argument('--eval', type=bool, default=False,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--train_seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', # batch size
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N', # hidden size
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1500000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--replay_buffer_type', type=str, default='prioritized') 

    # Validation
    parser.add_argument('--val_seed', type=int, default=5574, metavar='N',
                        help='random seed (default: 5574)')
    parser.add_argument('--validation', type=bool, default=True)
    parser.add_argument('--val_episodes_per_scene', type=int, default=50, metavar='N',
                        help='random seed (default: 50)')
    parser.add_argument('--val_episode_interval', type=int, default=200, metavar='N',
                        help='default: 100')
    parser.add_argument('--val_no_of_pedestrians', type=int, default=3, metavar='N',
                        help='default: 3')
    
    # Reward
    parser.add_argument('--goal_reward', type=int, default=20)
    parser.add_argument('--pedestrian_collision_reward', type=int, default=-20)

    parser.add_argument('--consider_all_waypoint', type=bool, default=False)

    parser.add_argument('--pca_reward', type=bool, default=True)
    parser.add_argument('--pca_reward_value', type=float, default=1)
    parser.add_argument('--pca_threshold', type=float, default=1)

    parser.add_argument('--waypoint_reach_threshold', type=float, default=0.1)

    # Frontier based navigation
    parser.add_argument('--frontier_based_navigation', type=bool, default=False)
    parser.add_argument('--frontier_selection_method', type=str, default="closest")
    parser.add_argument('--replan_steps', type=int, default=10)

    # Pedestrian
    parser.add_argument('--pedestrian_present', type=bool, default=True)
    parser.add_argument('--num_pedestrians', type=int, default=3)
    parser.add_argument('--ped_no_variable', type=bool, default=True) #PED VARIABLE
    parser.add_argument('--fixed_num_pedestrians', type=int, default=3)
    parser.add_argument('--highest_num_pedestrians', type=int, default=5)
    parser.add_argument('--orca_radius', type=float, default=0.5)
    parser.add_argument('--pedestrian_collision_threshold', type=float, default=0.3)
    parser.add_argument('--pedestrian_goal_threshold', type=float, default=0.3)
    parser.add_argument('--replan_if_collision', type=bool, default=False) # Replan
    parser.add_argument('--replan_map', type=str, default="Nogaussian") #Replanning type
    
    # Observations
    parser.add_argument('--obs_goal', type=bool, default=True)
    parser.add_argument('--obs_waypoints', type=bool, default=True)
    parser.add_argument('--obs_map', type=bool, default=True)
    parser.add_argument('--obs_previous_action', type=bool, default=True)
    parser.add_argument('--obs_pedestrian_map', type=bool, default=True)
    # parser.add_argument('--obs_map_lstm', type=bool, default=False)
    # parser.add_argument('--obs_lidar', type=bool, default=True)
    # parser.add_argument('--obs_replan', type=bool, default=False)
    parser.add_argument('--obs_pedestrian_pos', type=bool, default=True)

    parser.add_argument('--obs_normalized', type=bool, default=True)
    parser.add_argument('--obs_train', type=bool, default=True) #Obs encoder trained

    parser.add_argument('--map_encoder', type=str, default="cnn")
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--num_wps_input', type=int, default=5)

    # Checkpoint SAC
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/sac_checkpoint_fullTraining_fulldataset_raycast_obsTrain")
    parser.add_argument('--checkpoint_name', type=str, default="fullTraining_fulldataset_raycast_obsTrain")

    # Checkpoint Replay Buffer
    parser.add_argument('--load_checkpoint_memory', type=bool, default=False)
    parser.add_argument('--checkpoint_path_memory', type=str, default="checkpoints/sac_buffer_fullTraining_fulldataset_raycast_obsTrain_memory")
    parser.add_argument('--checkpoint_path_priority', type=str, default="checkpoints/sac_buffer_priority_fullTraining_fulldataset_raycast_obsTrain_memoryy")
    parser.add_argument('--checkpoint_name_memory', type=str, default="fullTraining_fulldataset_raycast_obsTrain_memory")
    

    # Env
    parser.add_argument('--env', type=str, default='simple')
    parser.add_argument('--env_type', type=str, default='with_map')
    parser.add_argument('--inflation_radius', type=float, default=2.5)
    parser.add_argument('--fov', type=float, default=90)
    parser.add_argument('--depth', type=float, default=5)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--map', type=str, default='raycast')
    parser.add_argument('--waypoint_interval', type=int, default=5)
    parser.add_argument('--robot_visible_to_pedestrians', type=bool, default=True) # Robot visible
    parser.add_argument('--episode_max_num_step', type=int, default=500)
    parser.add_argument('--hopper', type=bool, default=True)
    
    parser.add_argument('--train_continue', type=bool, default=False)
    parser.add_argument('--no_episode_trained', type=int, default=750)
    parser.add_argument('--no_ep_after_print', type=int, default=30)
    parser.add_argument('--scene_change_after_no_episode', type=int, default=100)
    parser.add_argument('--total_num_ep_scene', type=int, default=300)
    
    args = parser.parse_args()

    challenge = Challenge()
    challenge.submit(None, args)

if __name__ == "__main__":
    main()
