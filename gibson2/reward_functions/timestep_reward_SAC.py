from gibson2.utils.utils import l2_distance


class TimestepRewardSAC():
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        self.config = config
        self.timestep_reward = -0.001

    def get_reward(self, task, env):
        #print("timestep reward", self.timestep_reward)
        return self.timestep_reward
    
