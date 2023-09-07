from gibson2.utils.utils import l2_distance
from gibson2.reward_functions.reward_function_base import BaseRewardFunction
import time


class CollisionRewardSAC(BaseRewardFunction):
    """
    PedestrianCollision used for navigation tasks
    Episode terminates if the robot has collided with any pedestrian
    """

    def __init__(self, config):
        super(CollisionRewardSAC, self).__init__(config)
        self.pedestrian_collision_threshold = self.config.get(
            'pedestrian_collision_threshold', 0.3)
        self.pedestrian_collision_reward = -1

    def get_reward(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if the robot has collided more than self.max_collisions_allowed times

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = False
        robot_pos = env.robots[0].get_position()[:2]
        reward = 0
        for ped in task.pedestrians:
            ped_pos = ped.get_position()[:2]
            if l2_distance(robot_pos, ped_pos) < self.pedestrian_collision_threshold:
                done = True
                reward = self.pedestrian_collision_reward
                break
        #print("collision reward", reward)
        return reward
