from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance
import numpy as np
from gibson2.utils.utils import cartesian_to_polar

class OrientationRewardSAC(BaseRewardFunction):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative.
    """

    def __init__(self, config):
        super(OrientationRewardSAC, self).__init__(config)
        self.orientation_reward_weight = -0.01
        #self.waypoint_reach_threshold = self.config.get(
        #    'dist_tol', 0.3) # body width 0.36
        self.waypoint_reach_threshold = 0.1

    def get_reward(self, task, env):
        waypoint = env.waypoints[0]
        waypoint = np.array([waypoint[0], waypoint[1], 0])
        waypoint_robot_coord = env.task.global_to_local(env, waypoint)[:2]
        # waypoints in polar coordinate in robot frame
        waypoint_polar = np.array(cartesian_to_polar(waypoint_robot_coord[0],waypoint_robot_coord[1]))
        angle = abs(waypoint_polar[1])

        robot_pos = env.robots[0].get_position()[:2]
        waypoint = env.waypoints[0][:2]
        reward = 0.0
        if l2_distance(robot_pos, waypoint) > self.waypoint_reach_threshold:
            reward = self.orientation_reward_weight * angle
        #print("potential reward", reward)
        return reward


