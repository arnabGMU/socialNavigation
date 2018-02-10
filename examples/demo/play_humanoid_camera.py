from gibson.envs.humanoid_env import HumanoidNavigateEnv
from gibson.utils.play import play
import os
timestep = 1.0/(4 * 22)
frame_skip = 4
config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'humanoid_navigate.yaml')
print(config_file)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    env = HumanoidNavigateEnv(is_discrete = True, config = config_file)
    play(env, zoom=4, fps=int( 1.0/(timestep * frame_skip)))