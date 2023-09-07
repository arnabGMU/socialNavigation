from gibson2.utils.utils import l2_distance


class GoalRewardSAC():
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        self.config = config
        self.success_reward = 10
        #self.dist_tol = self.config.get('dist_tol', 0.5) #0.36 body width
        self.dist_tol = 0.1

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        success = l2_distance(
            env.robots[0].get_position()[:2],
            task.target_pos[:2]) <= self.dist_tol
        reward = self.success_reward if success else 0.0
        #print("goal reward", reward)
        return reward
