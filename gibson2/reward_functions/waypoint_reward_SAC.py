from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance

class WaypointRewardSAC(BaseRewardFunction):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative.
    """

    def __init__(self, config):
        super(WaypointRewardSAC, self).__init__(config)
        self.waypoint_reward = 0.1
        #self.waypoint_reach_threshold = self.config.get(
        #    'dist_tol', 0.3) # body width 0.36
        self.waypoint_reach_threshold = 0.1

    def get_reward(self, task, env):
        robot_pos = env.robots[0].get_position()[:2]
        waypoint = env.waypoints[0][:2]
        reward = 0.0
        if l2_distance(robot_pos, waypoint) <= self.waypoint_reach_threshold:
            reward = self.waypoint_reward
        #print("potential reward", reward)
        return reward


