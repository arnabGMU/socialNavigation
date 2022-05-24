import copy
import logging
import random
import time
from enum import IntEnum

import gym
import numpy as np
import pybullet as p

from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, BaseActionPrimitiveSet
from igibson.controllers import ControlType, JointController
from igibson.object_states.pose import Pose
from igibson.robots.manipulation_robot import IsGraspingState
from igibson.utils.motion_planning_utils import MotionPlanner
from igibson.utils.transform_utils import mat2euler, quat2mat

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

index_action_mapping = {
    0: "move",
    1: "pick",
    2: "place",
    3: "toggle",
    4: "pull",
    5: "push",
    6: "vis_pick",
    7: "vis_place",
    8: "vis_pull",
    9: "vis_push",
}

skill_object_offset_params = {
    0: {  # skill id: move
        "printer.n.03_1": [-0.7, 0, 0, 0],  # dx, dy, dz, target_yaw
        "table.n.02_1": [0, -0.6, 0, 0.5 * np.pi],
        # Pomaria_1_int, 2
        "hamburger.n.01_1": [0, -0.8, 0, 0.5 * np.pi],
        "hamburger.n.01_2": [0, -0.7, 0, 0.5 * np.pi],
        "hamburger.n.01_3": [0, -0.8, 0, 0.5 * np.pi],
        "ashcan.n.01_1": [0, 0.8, 0, -0.5 * np.pi],
        "countertop.n.01_1": [0.0, -0.8, 0, 0.5 * np.pi],
        # putting_away_Halloween_decorations
        "pumpkin.n.02_1": [0.4, 0.0, 0.0, 1.0 * np.pi],
        "pumpkin.n.02_2": [0, -0.5, 0, 0.5 * np.pi],
        "cabinet.n.01_1": [0.4, -1.15, 0, 0.5 * np.pi],
        # cleaning_microwave_oven
        # "sink.n.01_1-cleaning_microwave_oven": [0., -0.5, 0, 0.5 * np.pi],
        "sink.n.01_1-cleaning_microwave_oven": [0., -0.57, 0, 0.5 * np.pi],
        # "microwave.n.02_1-cleaning_microwave_oven": [0., -1.02, 0, 0.5 * np.pi],
        "microwave.n.02_1-cleaning_microwave_oven": [-0.2, -0.75, 0, 0.5 * np.pi],
        "cabinet.n.01_1-cleaning_microwave_oven": [1.0, -0.8, 0, 0.5 * np.pi],
    },
    1: {  # pick
        "printer.n.03_1": [-0.2, 0.0, 0.2],  # dx, dy, dz
        # Pomaria_1_int, 2
        "hamburger.n.01_1": [0.0, 0.0, 0.025],
        "hamburger.n.01_2": [0.0, 0.0, 0.025,],
        "hamburger.n.01_3": [0.0, 0.0, 0.025,],
        # putting_away_Halloween_decorations
        "pumpkin.n.02_1": [0.0, 0.0, 0.0, 1.0,],
        "pumpkin.n.02_2": [0.0, 0.0, 0.0, 1.0,],
        # cleaning_microwave_oven
        "towel": [0.0, 0.0, 0.0],
    },
    2: {  # place
        "table.n.02_1": [0, 0, 0.5],  # dx, dy, dz
        # Pomaria_1_int, 2
        # "ashcan.n.01_1": [0, 0, 0.5],
        # Ihlen_1_int, 0
        "ashcan.n.01_1": [0, 0, 0.5],
        # putting_away_Halloween_decorations
        "cabinet.n.01_1": [0.3, -0.55, 0.25],
        # cleaning_microwave_oven
        "sink.n.01_1-cleaning_microwave_oven": [-0.2, -0.08, 0.2],
        # "microwave.n.02_1-cleaning_microwave_oven": [-0.2, -0., -0.13],
        "microwave.n.02_1-cleaning_microwave_oven": [-0.35, -0., -0.4],
    },
    3: {  # toggle
        "printer.n.03_1": [-0.3, -0.25, 0.23],  # dx, dy, dz
        # cleaning_microwave_oven
        "sink.n.01_1-cleaning_microwave_oven": [-0.05, 0.18, 0.32],
        # "sink.n.01_1-cleaning_microwave_oven": [-0.05, 0.17, 0.32],
    },
    4: {  # pull
        "cabinet.n.01_1": [0.3, -0.55, 0.35, -1, 0, 0],  # dx, dy, dz
    },
    5: {  # push
        "cabinet.n.01_1": [0.3, -0.85, 0.35, 1, 0, 0],  # dx, dy, dz
    },
    6: {  # vis_pick
        "hamburger.n.01_1": [0, -0.8, 0, 0.5 * np.pi, 0.0, 0.0, 0.025],
        "hamburger.n.01_2": [0, -0.7, 0, 0.5 * np.pi, 0.0, 0.0, 0.025,],
        "hamburger.n.01_3": [0, -0.8, 0, 0.5 * np.pi, 0.0, 0.0, 0.025,],
        # vis: putting_away_Halloween_decorations
        "pumpkin.n.02_1": [0.4, 0.0, 0.0, 1.0 * np.pi, 0.0, 0.0, 0.025,],
        "pumpkin.n.02_2": [0, -0.5, 0, 0.5 * np.pi, 0.0, 0.0, 0.025,],
    },
    7: {  # vis_place
        "ashcan.n.01_1": [0, 0.8, 0, -0.5 * np.pi, 0, 0, 0.5],
        # vis: putting_away_Halloween_decorations
        "cabinet.n.01_1": [0.4, -1.15, 0, 0.5 * np.pi, 0.3, -0.60, 0.25],
    },
    8: {  # vis pull
        "cabinet.n.01_1": [0.3, -0.55, 0.35],  # dx, dy, dz
    },
    9: {  # vis push
        "cabinet.n.01_1": [0.3, -0.8, 0.35],  # dx, dy, dz
    },
}

action_list_installing_a_printer = [
    [0, "printer.n.03_1"],  # skill id, target_obj
    [1, "printer.n.03_1"],
    [0, "table.n.02_1"],
    [2, "table.n.02_1"],
    [3, "printer.n.03_1"],
]

action_list_throwing_away_leftovers_v1 = [
    [0, "hamburger.n.01_1"],
    [1, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [2, "ashcan.n.01_1"],  # place
    [0, "hamburger.n.01_2"],
    [1, "hamburger.n.01_2"],
    [0, "hamburger.n.01_3"],
    [1, "hamburger.n.01_3"],
]

action_list_throwing_away_leftovers_discrete = [
    [0, "countertop.n.01_1", 0],
    [0, "countertop.n.01_1", 1],
    [0, "countertop.n.01_1", 2],
    [6, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [7, "ashcan.n.01_1"],  # place
    [6, "hamburger.n.01_2"],
    [6, "hamburger.n.01_3"],
]

action_list_throwing_away_leftovers = [
    [0, "countertop.n.01_1", 0],
    [6, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [7, "ashcan.n.01_1"],  # place
    [6, "hamburger.n.01_2"],
    [6, "hamburger.n.01_3"],
]

action_list_putting_leftovers_away = [
    [0, "pasta.n.02_1"],
    [1, "pasta.n.02_1"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2"],
    [1, "pasta.n.02_2"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2_3"],
    [1, "pasta.n.02_2_3"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2_4"],
    [1, "pasta.n.02_2_4"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
]

# full set
action_list_putting_away_Halloween_decorations_v1 = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # pull
    [0, "pumpkin.n.02_1"],  # move
    [1, "pumpkin.n.02_1"],  # pick
    # [0, 'cabinet.n.01_1'],  # move
    [2, "cabinet.n.01_1"],  # place
    [0, "pumpkin.n.02_2"],  # move
    [1, "pumpkin.n.02_2"],  # pick
    # [0, 'cabinet.n.01_1'],  # move
    # [2, 'cabinet.n.01_1'],  # place
    [5, "cabinet.n.01_1"],  # push
]

# /home/robot/Desktop/behavior/iGibson-dev-jk/igibson/examples/robots/log_dir_his/20220510-001432_putting_away_Halloween_decorations_discrete_rgb_accumReward_m0.01
# wo vis operation
action_list_putting_away_Halloween_decorations_v2 = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # pull
    [0, "pumpkin.n.02_1"],  # move
    [1, "pumpkin.n.02_1"],  # pick
    [0, "cabinet.n.01_1"],  # move, repeated
    [2, "cabinet.n.01_1"],  # place
    [0, "pumpkin.n.02_2"],  # move
    [1, "pumpkin.n.02_2"],  # pick
    #
    # [0, 'cabinet.n.01_1'],  # move
    # [2, 'cabinet.n.01_1'],  # place
    #
    [5, "cabinet.n.01_1"],  # push
]  # * 4

# vis version: full sequence
action_list_putting_away_Halloween_decorations_v3 = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # vis pull
    [0, "pumpkin.n.02_1"],  # move
    [6, "pumpkin.n.02_1"],  # vis pick
    [0, "cabinet.n.01_1"],  # move
    [7, "cabinet.n.01_1"],  # vis place
    [0, "pumpkin.n.02_2"],  # move
    [6, "pumpkin.n.02_2"],  # vis pick
    [0, "cabinet.n.01_1"],  # move
    [7, "cabinet.n.01_1"],  # vis place
    [5, "cabinet.n.01_1"],  # vis push
]
# vis version: full set
action_list_putting_away_Halloween_decorations = [
    [0, "cabinet.n.01_1"],  # navigate_to 0
    [11, "cabinet.n.01_1"],  # pull 1
    [0, "pumpkin.n.02_1"],  # navigate_to 2
    [1, "pumpkin.n.02_1"],  # pick 3
    [2, "cabinet.n.01_1"],  # place 4
    [0, "pumpkin.n.02_2"],  # navigate_to 5
    [1, "pumpkin.n.02_2"],  # pick 6
    [5, "cabinet.n.01_1"],  # push 7
]

action_list_putting_away_Halloween_decorations_multi_discrete = [
    [0, "cabinet.n.01_1"],  # navigate_to 0
    [11, "cabinet.n.01_1"],  # pull 1
    [0, "pumpkin.n.02_1"],  # navigate_to 2
    [6, "pumpkin.n.02_1"],  # vis pick 3
    [7, "cabinet.n.01_1"],  # vis place 4
    [0, "pumpkin.n.02_2"],  # navigate_to 5
    [6, "pumpkin.n.02_2"],  # vis pick 6
    [5, "cabinet.n.01_1"],  # push 7
]

action_list_cleaning_microwave_oven = [
    [0, "cabinet.n.01_1"], # move 0
    [1, "towel"],  # pick 1
    [0, "sink.n.01_1"],  # move 2
    [2, "sink.n.01_1"],  # place 3
    [3, "sink.n.01_1"],  # toggle 4
    # [1, "towel"],  # pick
    [0, "microwave.n.02_1"],  # move 5
    [2, "microwave.n.02_1"],  # place 6
]

action_list_cleaning_microwave_oven_multi_discrete = action_list_cleaning_microwave_oven

action_list_room_rearrangement = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # vis pull
    [0, "pumpkin.n.02_1"],  # move
    [1, "pumpkin.n.02_1"],  # vis pick
    # [0, 'cabinet.n.01_1'],  # move
    [7, "cabinet.n.01_1"],  # vis place
    [0, "pumpkin.n.02_2"],  # move
    [6, "pumpkin.n.02_2"],  # vis pick
    # [0, 'cabinet.n.01_1'],  # move
    # [7, 'cabinet.n.01_1'],  # vis place
    [5, "cabinet.n.01_1"],  # vis push
]

action_dict = {
    "installing_a_printer": action_list_installing_a_printer,
    "throwing_away_leftovers": action_list_throwing_away_leftovers,
    "putting_leftovers_away": action_list_putting_leftovers_away,
    "putting_away_Halloween_decorations": action_list_putting_away_Halloween_decorations,
    "putting_away_Halloween_decorations_multi_discrete": action_list_putting_away_Halloween_decorations_multi_discrete,
    "throwing_away_leftovers_discrete": action_list_throwing_away_leftovers_discrete,
    "room_rearrangement": action_list_room_rearrangement,
    "cleaning_microwave_oven": action_list_cleaning_microwave_oven,
    "cleaning_microwave_oven_multi_discrete": action_list_cleaning_microwave_oven_multi_discrete,
}


class B1KActionPrimitive(IntEnum):
    NAVIGATE_TO = 0
    PICK = 1
    PLACE = 2
    TOGGLE = 3
    PULL = 4
    PUSH = 5
    VIS_PICK = 6
    VIS_PLACE = 7
    DUMMY = 10
    PUSH_OPEN = 11


class B1KActionPrimitives(BaseActionPrimitiveSet):
    def __init__(self, env, task, scene, robot, arm=None, execute_free_space_motion=False, action_space_type='discrete'):
        """ """
        super().__init__(env, task, scene, robot)
        self.action_space_type = action_space_type

        if arm is None:
            self.arm = self.robot.default_arm
        else:
            self.arm = arm
        logger.info("Using with the default arm: {}".format(self.arm))

        # Checks for the right type of robot and controller to use planned trajectories
        assert robot.grasping_mode == "assisted", "This APs implementation requires assisted grasping to check success"
        if robot.model_name in ["Tiago", "Fetch"]:
            assert not robot.rigid_trunk, "The APs will use the trunk of Fetch/Tiago, it can't be rigid"
        assert isinstance(
            robot._controllers["arm_" + self.arm], JointController
        ), "The arm to use with the primitives must be controlled in joint space"
        assert (
            robot._controllers["arm_" + self.arm].control_type == ControlType.POSITION
        ), "The arm to use with the primitives must be controlled in absolute positions"
        assert not robot._controllers[
            "arm_" + self.arm
        ].use_delta_commands, "The arm to use with the primitives cannot be controlled with deltas"

        self.controller_functions = {
            B1KActionPrimitive.NAVIGATE_TO: self._navigate_to,
            B1KActionPrimitive.PICK: self._pick,
            B1KActionPrimitive.PLACE: self._place,
            B1KActionPrimitive.TOGGLE: self._toggle,
            B1KActionPrimitive.PULL: self._pull,
            B1KActionPrimitive.PUSH: self._push,
            B1KActionPrimitive.VIS_PICK: self._vis_pick,
            B1KActionPrimitive.VIS_PLACE: self._vis_place,
            B1KActionPrimitive.PUSH_OPEN: self._push_open,
            B1KActionPrimitive.DUMMY: self._dummy,
        }

        print('self.env.config["task"]: ', self.env.config["task"])
        if self.env.config["task"] == "throwing_away_leftovers":
            self.action_list = action_dict["throwing_away_leftovers_discrete"]
            skill_object_offset_params[0]["countertop.n.01_1"] = [
                [0.0, -0.8, 0, 0.1 * np.pi],
                [0.0, -0.8, 0, 0.5 * np.pi],
                [0.0, -0.8, 0, 0.8 * np.pi],
            ]
        else:
            if self.action_space_type == 'multi_discrete':
                self.action_list = action_dict[self.env.config["task"]+'_multi_discrete']
            else:
                self.action_list = action_dict[self.env.config["task"]]
        self.num_discrete_action = len(self.action_list)
        self.initial_pos_dict = {}  # TODO: what for?
        full_observability_2d_planning = True
        collision_with_pb_2d_planning = True
        self.planner = MotionPlanner(
            self.env,
            optimize_iter=10,
            full_observability_2d_planning=full_observability_2d_planning,
            collision_with_pb_2d_planning=collision_with_pb_2d_planning,
            visualize_2d_planning=False,
            visualize_2d_result=False,
            fine_motion_plan=False,
        )
        self.default_direction = np.array((0.0, 0.0, -1.0))  # default hit normal
        self.execute_free_space_motion = execute_free_space_motion

        # Whether we check if the objects have moved from the previous navigation attempt and do not try to navigate
        # to them if they have (this avoids moving to objects after pick and place)
        self.obj_pose_check = True
        self.task_obj_list = self.env.task.object_scope
        self.print_log = True
        self.skip_base_planning = True
        self.skip_arm_planning = True  # False
        self.is_grasping = False
        self.fast_execution = False
        self.action_space_type = action_space_type

    def get_action_space(self):
        if self.action_space_type == 'discrete':
            return gym.spaces.Discrete(self.num_discrete_action)
        if self.action_space_type == 'multi_discrete':
            if self.env.config['task'] in ['throwing_away_leftovers']:
                multi_discrete_grid_size = 10
                return gym.spaces.MultiDiscrete([self.num_discrete_action] + [multi_discrete_grid_size, ] * 1)
            elif self.env.config['task'] in ['putting_away_Halloween_decorations']:
                multi_discrete_grid_size = 4
                return gym.spaces.MultiDiscrete([self.num_discrete_action] + [multi_discrete_grid_size, ] * 1)
            elif self.env.config['task'] in ['cleaning_microwave_oven']:
                multi_discrete_grid_size = 2
                return gym.spaces.MultiDiscrete([self.num_discrete_action] + [multi_discrete_grid_size, ] * 1)

    def apply(self, action_index, obs=None):
        if self.action_space_type == 'discrete':
            if action_index == 10:
                return self._dummy()
            else:
                primitive_obj_pair = self.action_list[action_index]
                # if self.env.config['task'] in ['cleaning_microwave_oven'] and not (primitive_obj_pair[1] in ['towel']):
                #     return self.controller_functions[primitive_obj_pair[0]](primitive_obj_pair[1] + '-' + self.env.config['task'], obs)
                # else:
                return self.controller_functions[primitive_obj_pair[0]](primitive_obj_pair[1], obs)

        elif self.action_space_type == 'multi_discrete':
            # print('action_index: ', action_index, action_index[0])
            if action_index[0] == 10:
                return self._dummy()
            else:
                primitive_obj_pair = self.action_list[action_index[0]]
                if self.env.config['task'] in ['throwing_away_leftovers']:
                    if primitive_obj_pair[0] in [0, ] and primitive_obj_pair[1] == 'countertop.n.01_1':
                        # [1, 10] * [1, pi]
                        yaw = (action_index[1] + 1) * 0.1 * np.pi
                        print('action: {} {}, discrete action: {}'.format(index_action_mapping[primitive_obj_pair[0]],
                                                                         primitive_obj_pair[1], action_index[1] + 1))
                    else:
                        yaw = None
                elif self.env.config['task'] in ['putting_away_Halloween_decorations']:
                    if primitive_obj_pair[0] in [0, ] and primitive_obj_pair[1] in ['pumpkin.n.02_1', 'pumpkin.n.02_2']:
                        yaw = (action_index[1] + 1) * 0.5 * np.pi
                        # print('\n\n\n\n\n', primitive_obj_pair, 'yaw', yaw)
                    else:
                        yaw = None
                elif self.env.config['task'] in ['cleaning_microwave_oven']:
                    if primitive_obj_pair[0] in [1, 2]:
                        yaw = action_index[1]
                    else:
                        yaw = None
                else:
                    yaw = None
                return self.controller_functions[primitive_obj_pair[0]](primitive_obj_pair[1], obs, yaw=yaw)

    def _get_obj_in_hand(self):
        obj_in_hand_id = self.robot._ag_obj_in_hand[self.arm]  # TODO(MP): Expose this interface.
        obj_in_hand = self.scene.objects_by_id[obj_in_hand_id] if obj_in_hand_id is not None else None
        return obj_in_hand

    def _get_still_action(self):
        # The camera and arm(s) and any other absolution joint position controller will be controlled with absolute positions
        state = self.robot.get_joint_states()
        joint_positions = np.array([state[joint_name][0] for joint_name in state])
        action = np.zeros(self.robot.action_dim)
        for controller_name, controller in self.robot._controllers.items():
            if (
                isinstance(controller, JointController)
                and controller.control_type == ControlType.POSITION
                and not controller.use_delta_commands
                and not controller.use_constant_goal_position
            ):
                action_idx = self.robot.controller_action_idx[controller_name]
                joint_idx = self.robot.controller_joint_idx[controller_name]
                logger.debug(
                    "Setting action to absolute position for {} in action dims {} corresponding to joint idx {}".format(
                        controller_name, action_idx, joint_idx
                    )
                )
                action[action_idx] = joint_positions[joint_idx]
        if self.is_grasping:
            # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
            action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1.0
        return action

    def _execute_ee_path(
        self, path, ignore_failure=False, stop_on_contact=False, reverse_path=False, while_grasping=False
    ):
        for arm_action in path if not reverse_path else reversed(path):
            if stop_on_contact and len(self.robot._find_gripper_contacts(arm=self.arm)[0]) != 0:
                logger.info("Contact detected. Stop motion")
                logger.debug("Contacts {}".format(self.robot._find_gripper_contacts(arm=self.arm)))
                logger.debug("Finger ids {}".format([link.link_id for link in self.robot.finger_links[self.arm]]))
                return
            logger.debug("Executing action {}".format(arm_action))
            full_body_action = self._get_still_action()
            # This assumes the arms are called "arm_"+self.arm. Maybe some robots do not follow this convention
            arm_controller_action_idx = self.robot.controller_action_idx["arm_" + self.arm]
            full_body_action[arm_controller_action_idx] = arm_action

            if while_grasping:
                # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
                full_body_action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1.0
            yield full_body_action

        if stop_on_contact:
            logger.warning("End of motion and no contact detected")
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.EXECUTION_ERROR, "No contact was made.")
        else:
            logger.debug("End of the path execution")

    def _execute_grasp(self):
        get_still_action_time = time.time()
        action = self._get_still_action()
        after_get_still_action_time = time.time()
        # print('_get_still_action: {}'.format(after_get_still_action_time - get_still_action_time))
        # TODO: Extend to non-binary grasping controllers
        # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
        action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1.0

        grasping_steps = 5 if self.fast_execution else 9 # 10
        for _ in range(grasping_steps):
            yield action
        yield_action_time = time.time()
        # print('yield grasping action: {}'.format(yield_action_time - after_get_still_action_time))
        grasped_object = self._get_obj_in_hand()
        if grasped_object is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "No object detected in hand after executing grasp.",
            )
        else:
            logger.info("Execution of grasping ended with grasped object {}".format(grasped_object.name))
            self.is_grasping = True

    def _execute_ungrasp(self):
        action = self._get_still_action()

        # TODO: Extend to non-binary grasping controllers
        # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
        ungrasping_steps = 5 if self.fast_execution else 2 # 10
        after_get_still_action_time = time.time()
        for idx in range(ungrasping_steps):
            action[self.robot.controller_action_idx["gripper_" + self.arm]] = 0.0
            yield action
        yield_action_time = time.time()
        # print('yield ungrasping action: {}'.format(yield_action_time - after_get_still_action_time))
        if self._get_obj_in_hand() is not None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Object still detected in hand after executing release.",
                {"object_in_hand": self._get_obj_in_hand()},
            )
        self.is_grasping = False

    def _dummy(self):
        logger.info("Dummy".format())
        yield self._get_still_action()
        logger.info("Finished dummy".format())

    def _navigate_to(self, object_name, obs=None, yaw=None):
        logger.info("Navigating to object {}".format(object_name))
        if self.env.config['task'] in ['cleaning_microwave_oven'] and not (object_name in ['towel']):
            params = skill_object_offset_params[B1KActionPrimitive.NAVIGATE_TO][object_name+'-'+self.env.config['task']]
        else:
            params = skill_object_offset_params[B1KActionPrimitive.NAVIGATE_TO][object_name]

        # If we check whether the object has moved from its initial location. If we check that, and the object has moved
        # more than a threshold, we ignore the command
        moved_distance_threshold = 1e-1
        if self.obj_pose_check:
            if self.env.config["task"] in ["putting_away_Halloween_decorations"]:
                obj_pos = self.env.task.object_scope[object_name].states[Pose].get_value()[0]
                if object_name in ["pumpkin.n.02_1", "pumpkin.n.02_2"]:
                    if object_name not in self.initial_pos_dict:
                        self.initial_pos_dict[object_name] = obj_pos
                    else:
                        moved_distance = np.abs(np.sum(self.initial_pos_dict[object_name] - obj_pos))
                        if moved_distance > moved_distance_threshold:
                            raise ActionPrimitiveError(
                                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                                "Object moved from its initial location",
                                {"object_to_navigate": object_name},
                            )

        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        # acquire the base direction
        euler = mat2euler(mat)
        if yaw is None:
            target_yaw = euler[-1] + params[3]
        else:
            print('navigate to yaw: ', yaw)
            target_yaw = euler[-1] + yaw

        obj_idx_to_ignore = []
        if self.is_grasping:
            obj_idx_to_ignore = [self.robot._ag_obj_in_hand[self.arm]]

        plan = self.planner.plan_base_motion(
            [obj_pos[0] + vector[0], obj_pos[1] + vector[1], target_yaw],
            plan_full_base_motion=not self.skip_base_planning,
            obj_idx_to_ignore=obj_idx_to_ignore,
        )

        if plan is not None and len(plan) > 0:
            self.planner.visualize_base_path(plan, keep_last_location=True)
        else:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No base path found to object",
                {"object_to_navigate": object_name},
            )
        yield self._get_still_action()
        logger.info("Finished navigating to object: {}".format(object_name))
        # return

    def _pick(self, object_name, obs=None, yaw=None):
        logger.info("Picking object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PICK][object_name]
        if object_name in ['towel'] and self.env.config['task'] in ['cleaning_microwave_oven']:
            # towel: 132
            # plt.imshow(self.state['ins_seg'])
            # plt.show()
            object_id = 132
            pos, orn = p.getBasePositionAndOrientation(object_id)  # towel: 132
            obj_pos = list(pos)
            obj_rot_XYZW = list(orn)
        else:
            obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
            object_id = self.task_obj_list[object_name].get_body_ids()[0]  # Assume single body objects

        # Don't do anything if the object is already grasped.
        robot_is_grasping = self.robot.is_grasping(candidate_obj=None)
        robot_is_grasping_obj = self.robot.is_grasping(candidate_obj=object_id)
        if robot_is_grasping == IsGraspingState.TRUE:
            if robot_is_grasping_obj == IsGraspingState.TRUE:
                logger.warning("Robot already grasping the desired object")
                yield np.zeros(self.robot.action_dim)
                return
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "Cannot grasp when hand is already full.",
                    {"object": object_name},
                )

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        pre_grasping_distance = 0.0
        plan_full_pre_grasp_motion = not self.skip_arm_planning

        if yaw == 0:
            picking_direction = np.array((1.0, -0.0, 0.0))  # from right side, euler angles:  x: 0.5235988, y: 0, z: 1.5707963
            # axis with angle magnitude: 0.4103802, -0.4103802, 1.5315599
        elif yaw == 1:
            picking_direction = np.array((0.0, 0.0, -1.0))
        else:
            picking_direction = self.default_direction

        if len(params) > 3:
            logger.warning(
                "The number of params indicate that this picking position was made robot-agnostic."
                "Adding finger offset."
            )
            finger_size = self.robot.finger_lengths[self.arm]
            pick_place_pos -= picking_direction * finger_size

        pre_pick_path, interaction_pick_path = self.planner.plan_ee_pick(
            pick_place_pos,
            grasping_direction=picking_direction,
            pre_grasping_distance=pre_grasping_distance,
            plan_full_pre_grasp_motion=plan_full_pre_grasp_motion,
        )

        if (
            pre_pick_path is None
            or len(pre_pick_path) == 0
            or interaction_pick_path is None
            or (len(interaction_pick_path) == 0 and pre_grasping_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to pick object",
                {"object_to_pick": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Visualizing pre-pick path")
        self.planner.visualize_arm_path(pre_pick_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()
        if pre_grasping_distance != 0:
            # Then, execute the interaction_pick_path stopping if there is a contact
            logger.info("Executing interaction-pick path")
            yield from self._execute_ee_path(interaction_pick_path, stop_on_contact=True)
        # At the end, close the hand
        logger.info("Executing grasp")
        yield from self._execute_grasp()
        if pre_grasping_distance != 0:
            logger.info("Executing retracting path")
            yield from self._execute_ee_path(
                interaction_pick_path, stop_on_contact=False, reverse_path=True, while_grasping=True
            )
        logger.info("Executing retracting path")
        if plan_full_pre_grasp_motion:
            self.planner.visualize_arm_path(
                pre_pick_path, arm=self.arm, reverse_path=True, grasped_obj_id=object_id, keep_last_location=True
            )
        else:
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
                grasped_obj_id=object_id,
            )
        yield self._get_still_action()
        logger.info("Pick action completed")

    def _place(self, object_name, obs=None, yaw=None):
        logger.info("Placing on object {}".format(object_name))
        if self.env.config['task'] in ['cleaning_microwave_oven'] and not (object_name in ['towel']):
            params = skill_object_offset_params[B1KActionPrimitive.PLACE][object_name+'-'+self.env.config['task']]
        else:
            params = skill_object_offset_params[B1KActionPrimitive.PLACE][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        np_array = np.array(params[:3])
        np_array[0] += random.uniform(0, 0.2)
        vector = mat @ np_array

        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        plan_full_pre_drop_motion = not self.skip_arm_planning
        print('place yaw: ', yaw)
        if yaw == 0:
            ee_dropping_orn = np.array([1.0, 0.0, 0.0])
        elif yaw == 1:
            ee_dropping_orn = np.array([0.0, 0.0, -1.0])
        else:
            ee_dropping_orn = np.array([0.0, 0.0, -1.0])
        pre_drop_path, _ = self.planner.plan_ee_drop(
            pick_place_pos, arm=self.arm, plan_full_pre_drop_motion=plan_full_pre_drop_motion,
            ee_dropping_orn=ee_dropping_orn,
        )

        if pre_drop_path is None or len(pre_drop_path) == 0:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to place object",
                {"object_to_place": object_name},
            )

        # First, teleport the robot to the pre-ungrasp motion
        logger.info("Visualizing pre-place path")
        self.planner.visualize_arm_path(
            pre_drop_path,
            arm=self.arm,
            grasped_obj_id=self.robot._ag_obj_in_hand[self.arm],
            keep_last_location=True,
        )
        yield self._get_still_action()
        # At the end, open the hand
        logger.info("Executing ungrasp")
        yield from self._execute_ungrasp()
        # Then, retract the arm
        logger.info("Executing retracting path")
        if plan_full_pre_drop_motion:  # Visualizing the full path...
            self.planner.visualize_arm_path(
                pre_drop_path,
                arm=self.arm,
                keep_last_location=True,
                reverse_path=True,
            )
        else:  # ... or directly teleporting to the last location
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        logger.info("Place action completed")

    def _toggle(self, object_name, obs=None, yaw=None):
        logger.info("Toggling object {}".format(object_name))
        if self.env.config['task'] in ['cleaning_microwave_oven'] and not (object_name in ['towel']):
            params = skill_object_offset_params[B1KActionPrimitive.TOGGLE][object_name+'-'+self.env.config['task']]
        else:
            params = skill_object_offset_params[B1KActionPrimitive.TOGGLE][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        toggle_pos = copy.deepcopy(obj_pos)
        toggle_pos[0] += vector[0]
        toggle_pos[1] += vector[1]
        toggle_pos[2] += vector[2]

        pre_toggling_distance = 0.0
        plan_full_pre_toggle_motion = not self.skip_arm_planning

        pre_toggle_path, toggle_interaction_path = self.planner.plan_ee_toggle(
            toggle_pos,
            -np.array(self.default_direction),
            pre_toggling_distance=pre_toggling_distance,
            plan_full_pre_toggle_motion=plan_full_pre_toggle_motion,
        )

        if (
            pre_toggle_path is None
            or len(pre_toggle_path) == 0
            or toggle_interaction_path is None
            or (len(toggle_interaction_path) == 0 and pre_toggling_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to toggle object",
                {"object_to_toggle": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Visualizing pre-toggle path")
        self.planner.visualize_arm_path(pre_toggle_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.info("Executing interaction-toggle path")
        yield from self._execute_ee_path(toggle_interaction_path, stop_on_contact=True)

        logger.info("Executing retracting path")
        if plan_full_pre_toggle_motion:
            self.planner.visualize_arm_path(pre_toggle_path, arm=self.arm, reverse_path=True, keep_last_location=True)
        else:
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()
        yield self._get_still_action()

        logger.info("Toggle action completed")

    def _pull(self, object_name, obs=None, yaw=None):
        logger.info("Pulling object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PULL][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        pulling_pos = copy.deepcopy(obj_pos)
        pulling_pos[0] += vector[0]
        pulling_pos[1] += vector[1]
        pulling_pos[2] += vector[2]

        pulling_direction = np.array(params[3:6])
        ee_pulling_orn = p.getQuaternionFromEuler((np.pi / 2, 0, 0))
        pre_pulling_distance = 0.10
        pulling_distance = 0.30

        finger_size = self.robot.finger_lengths[self.arm]
        pulling_pos += pulling_direction * finger_size

        plan_full_pre_pull_motion = not self.skip_arm_planning

        pre_pull_path, approach_interaction_path, pull_interaction_path = self.planner.plan_ee_pull(
            pulling_location=pulling_pos,
            pulling_direction=pulling_direction,
            ee_pulling_orn=ee_pulling_orn,
            pre_pulling_distance=pre_pulling_distance,
            pulling_distance=pulling_distance,
            plan_full_pre_pull_motion=plan_full_pre_pull_motion,
            pulling_steps=15,
        )

        if (
            pre_pull_path is None
            or len(pre_pull_path) == 0
            or approach_interaction_path is None
            or (len(approach_interaction_path) == 0 and pre_pulling_distance != 0)
            or pull_interaction_path is None
            or (len(pull_interaction_path) == 0 and pulling_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to pull object",
                {"object_to_pull": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Visualizing pre-pull path")
        self.planner.visualize_arm_path(pre_pull_path, arm=self.arm)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.info("Executing approaching pull path")
        yield from self._execute_ee_path(approach_interaction_path, stop_on_contact=True)
        # At the end, close the hand
        logger.info("Executing grasp")
        yield from self._execute_grasp()
        # Then, execute the interaction_pull_path
        # Since we may have stopped earlier due to contact, the precomputed path may be wrong
        # We have two options here: 1) (re)plan from the current pose (online planning), or 2) find the closest point
        # in the precomputed trajectory and start the execution there. Implementing option 1)
        logger.info("Replaning interaction pull path and executing")
        current_ee_position = self.robot.get_eef_position(arm=self.arm)
        current_ee_orn = self.robot.get_eef_orientation(arm=self.arm)
        state = self.robot.get_joint_states()
        joint_positions = np.array([state[joint_name][0] for joint_name in state])
        arm_joint_positions = joint_positions[self.robot.controller_joint_idx["arm_" + self.arm]]
        pull_interaction_path = self.planner.plan_ee_straight_line_motion(
            arm_joint_positions,
            current_ee_position,
            pulling_direction,
            ee_orn=current_ee_orn,
            line_length=pulling_distance,
            arm=self.arm,
        )
        yield from self._execute_ee_path(pull_interaction_path, while_grasping=True)
        # Then, open the hand
        logger.info("Executing ungrasp")
        yield from self._execute_ungrasp()
        logger.info("Untuck arm")
        self.planner.visualize_arm_path(
            [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
            arm=self.arm,
            keep_last_location=True,
        )
        yield self._get_still_action()

        logger.info("Pull action completed")

    def _pull_open_v2(self, object_name, obs=None, yaw=None):
        logger.info("Pulling object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PULL][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        # acquire the base direction
        euler = mat2euler(mat)
        target_yaw = euler[-1] + params[3]
        print('target_yaw: ', target_yaw, params)
        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        obj_idx_to_ignore = []
        if self.is_grasping:
            obj_idx_to_ignore = [self.robot._ag_obj_in_hand[self.arm]]

        # plan = self.planner.plan_arm_pull(pick_place_pos, hit_normal=np.array((-1.0, 0.0, 0.0)))
        # self.planner.execute_arm_pull(plan, pick_place_pos, -np.array(hit_normal))
        plan_ee_pull_time = time.time()
        # plan = self.planner.plan_arm_pull(pick_place_pos, np.array((-1.0, 0.0, 0.0)))
        plan = self.planner.plan_base_motion(
            [obj_pos[0] + vector[0], obj_pos[1] + vector[1], target_yaw],
            plan_full_base_motion=not self.skip_base_planning,
            obj_idx_to_ignore=obj_idx_to_ignore,
        )
        print('plan: ', plan)

        pre_pull_time = time.time()
        print('plan ee pull time: {}'.format(pre_pull_time - plan_ee_pull_time))
        # print("Visualizing pre-pull path")
        # print('pre_pull_path: ', pre_pull_path, len(pre_pull_path))
        # self.planner.visualize_arm_path(pre_pull_path, arm=self.arm)
        # yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        # print("Executing approaching pull path")
        # print('approach_interaction_path: ', approach_interaction_path, len(approach_interaction_path), np.sum(approach_interaction_path, axis=0).shape)
        # sum_approach_interaction_path = [np.sum(approach_interaction_path, axis=0)]
        # print('sum_approach_interaction_path: ', sum_approach_interaction_path)
        start_execute_ee_time = time.time()
        print('pre_pull_time: {}'.format(start_execute_ee_time - pre_pull_time))
        yield from self._execute_ee_path(plan, stop_on_contact=True)
        execute_ee_time = time.time()
        print('execute_ee_time: {}'.format(execute_ee_time - start_execute_ee_time))
        # At the end, close the hand
        # print("Executing grasp")
        yield from self._execute_grasp()
        grasp_time = time.time()
        print('grasp_time: {}'.format(grasp_time - execute_ee_time))
        yield from self._execute_ee_path(plan[::-1], while_grasping=True)
        pull_interaction_time = time.time()
        print('pull_interaction_time: {}'.format(pull_interaction_time - grasp_time))
        # Then, open the hand
        # print("Executing ungrasp")
        yield from self._execute_ungrasp()
        ungrasp_time = time.time()
        print('ungrasp_time: {}'.format(ungrasp_time - pull_interaction_time))
        yield self._get_still_action()
        still_action_time = time.time()
        print('still_action_time: {}'.format(still_action_time - ungrasp_time))
        print('total time: {}'.format(still_action_time - plan_ee_pull_time))
        print("Pull action completed")

    def _pull_open_v3(self, object_name, obs=None, yaw=None):
        logger.info("Pulling object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PULL][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        # acquire the base direction
        euler = mat2euler(mat)
        target_yaw = euler[-1] + params[3]
        print('target_yaw: ', target_yaw, params)
        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        body_joint_pairs = []
        for obj in self.env.scene.objects_by_category['bottom_cabinet']:
            # print('obj: ', obj)
            for body_id in obj.get_body_ids():
                for joint_id in range(p.getNumJoints(body_id)):
                    # cache current physics state
                    state_id = p.saveState()
                    all_joint_info = p.getJointInfo(body_id, joint_id)
                    print('body_id: {}, joint_id: {}, all_joint_info: {}, all_joint_info.shape: {}'
                          .format(body_id, joint_id, all_joint_info, len(all_joint_info)))  # , 17
                    j_low, j_high = p.getJointInfo(body_id, joint_id)[8:10]
                    j_type = p.getJointInfo(body_id, joint_id)[2]
                    parent_idx = p.getJointInfo(body_id, joint_id)[-1]
                    if j_type not in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                        p.removeState(state_id)
                        continue
                    # this is the continuous joint
                    if j_low >= j_high:
                        p.removeState(state_id)
                        continue
                    # this is the 2nd degree joint, ignore for now
                    if parent_idx != -1:
                        p.removeState(state_id)
                        continue

                    p.resetJointState(body_id, joint_id, 0.3)

                    body_joint_pairs.append((body_id, joint_id))
        print('body_joint_pairs: ', body_joint_pairs)
        print("Pull action completed")

    def _pull_open_v4(self, object_name, obs=None, yaw=None):
        logger.info("Pulling object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PULL][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        pulling_direction = np.array(params[3:6])
        ee_pulling_orn = p.getQuaternionFromEuler((np.pi / 2, 0, 0))
        pre_pulling_distance = 0.10
        pulling_distance = 0.30

        plan_full_pre_pull_motion = not self.skip_arm_planning
        plan_ee_pull_time = time.time()
        pre_pull_path, approach_interaction_path, pull_interaction_path = self.planner.plan_ee_pull(
            pulling_location=pick_place_pos,
            pulling_direction=pulling_direction,
            ee_pulling_orn=ee_pulling_orn,
            pre_pulling_distance=pre_pulling_distance,
            pulling_distance=pulling_distance,
            plan_full_pre_pull_motion=plan_full_pre_pull_motion,
            pulling_steps=4,
        )

        if (
                pre_pull_path is None
                or len(pre_pull_path) == 0
                or approach_interaction_path is None
                or (len(approach_interaction_path) == 0 and pre_pulling_distance != 0)
                or pull_interaction_path is None
                or (len(pull_interaction_path) == 0 and pulling_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to pull object",
                {"object_to_pull": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        pre_pull_time = time.time()
        print('plan ee pull time: {}'.format(pre_pull_time - plan_ee_pull_time))
        # print("Visualizing pre-pull path")
        # print('pre_pull_path: ', pre_pull_path, len(pre_pull_path))
        self.planner.visualize_arm_path(pre_pull_path, arm=self.arm)
        # yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        # print("Executing approaching pull path")
        # print('approach_interaction_path: ', approach_interaction_path, len(approach_interaction_path), np.sum(approach_interaction_path, axis=0).shape)
        # sum_approach_interaction_path = [np.sum(approach_interaction_path, axis=0)]
        # print('sum_approach_interaction_path: ', sum_approach_interaction_path)
        start_execute_ee_time = time.time()
        # print('pre_pull_time: {}'.format(start_execute_ee_time - pre_pull_time))
        yield from self._execute_ee_path(approach_interaction_path, stop_on_contact=True)
        execute_ee_time = time.time()
        # print('execute_ee_time: {}'.format(execute_ee_time - start_execute_ee_time))
        # At the end, close the hand
        # print("Executing grasp")
        yield from self._execute_grasp()
        grasp_time = time.time()
        # print('grasp_time: {}'.format(grasp_time - execute_ee_time))
        yield from self._execute_ee_path(pull_interaction_path, while_grasping=True)
        pull_interaction_time = time.time()
        # print('pull_interaction_time: {}'.format(pull_interaction_time - grasp_time))
        # Then, open the hand
        # print("Executing ungrasp")
        yield from self._execute_ungrasp()
        ungrasp_time = time.time()
        # print('ungrasp_time: {}'.format(ungrasp_time - pull_interaction_time))
        yield self._get_still_action()
        still_action_time = time.time()
        # print('still_action_time: {}'.format(still_action_time - ungrasp_time))
        # print('total time: {}'.format(still_action_time - plan_ee_pull_time))
        print("Pull action completed")

    def _push(self, object_name, obs=None, yaw=None):
        logger.info("Pushing object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PUSH][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        pushing_direction = np.array(params[3:6])

        pushing_distance = 0.2
        ee_pushing_orn = np.array(p.getQuaternionFromEuler((0, np.pi / 2, 0)))

        plan_full_pre_push_motion = not self.skip_arm_planning

        pre_push_path, push_interaction_path = self.planner.plan_ee_push(
            pick_place_pos,
            pushing_direction=pushing_direction,
            ee_pushing_orn=ee_pushing_orn,
            pushing_distance=pushing_distance,
            plan_full_pre_push_motion=plan_full_pre_push_motion,
            pushing_steps=5,
        )

        if (
            pre_push_path is None
            or len(pre_push_path) == 0
            or push_interaction_path is None
            or (len(push_interaction_path) == 0 and pushing_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to push object",
                {"object_to_push": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Pre-push motion")
        self.planner.visualize_arm_path(pre_push_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.info("Pushing interaction")
        yield from self._execute_ee_path(push_interaction_path, stop_on_contact=False)

        logger.info("Tuck arm")
        self.planner.visualize_arm_path(
            [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
            arm=self.arm,
            keep_last_location=True,
        )
        yield self._get_still_action()

        logger.info("Push action completed")

    def _push_open(self, object_name, obs=None, yaw=None):
        logger.info("Pushing object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PULL][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]
        is_grasping = self.robot.is_grasping() == True
        # print('is_grasping: ', is_grasping)

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        pushing_direction = np.array(params[3:6])

        pushing_distance = 0.2
        ee_pushing_orn = np.array(p.getQuaternionFromEuler((0, np.pi / 2, 0)))

        plan_full_pre_push_motion = not self.skip_arm_planning

        pre_push_path, push_interaction_path = self.planner.plan_ee_push(
            pick_place_pos,
            pushing_direction=pushing_direction,
            ee_pushing_orn=ee_pushing_orn,
            pushing_distance=pushing_distance,
            plan_full_pre_push_motion=plan_full_pre_push_motion,
            pushing_steps=5,
        )

        if (
                pre_push_path is None
                or len(pre_push_path) == 0
                or push_interaction_path is None
                or (len(push_interaction_path) == 0 and pushing_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to push object",
                {"object_to_push": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Pre-push motion")
        if is_grasping:
            self.planner.visualize_arm_path(pre_push_path, arm=self.arm, keep_last_location=True, grasped_obj_id=self.robot._ag_obj_in_hand[self.arm])
        else:
            self.planner.visualize_arm_path(pre_push_path, arm=self.arm, keep_last_location=True,)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.info("Pushing interaction")
        yield from self._execute_ee_path(push_interaction_path, stop_on_contact=False)

        logger.info("Tuck arm")
        if is_grasping:
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
                grasped_obj_id=self.robot._ag_obj_in_hand[self.arm],
            )
        else:
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )
        yield self._get_still_action()
        # yield from self._place(object_name)  # cabinet
        logger.info("Push action completed")

    def _vis_pick(self, object_name, obs=None, yaw=None):
        logger.info("Vis_pick object {}".format(object_name))
        ins_seg = np.round(obs['ins_seg'][:, :, 0]).astype(int)
        id_ins_seg = self.env.simulator.renderer.get_pb_ids_for_instance_ids(ins_seg)
        obj_id = self.task_obj_list[object_name].get_body_ids()
        print('obj_id in id_ins_seg: ', obj_id in id_ins_seg)
        if obj_id in id_ins_seg:
            yield from self._navigate_to(object_name, yaw=yaw)
            yield from self._pick(object_name)
            print('Vis_pick {} completed'.format(object_name, ))
        else:
            yield self._get_still_action()
            print('Vis_pick {} ignored'.format(object_name, ))

    def _vis_place(self, object_name, obs=None, yaw=None):
        logger.info("Vis_place Pushing object {}".format(object_name))
        ins_seg = np.round(obs['ins_seg'][:, :, 0]).astype(int)
        id_ins_seg = self.env.simulator.renderer.get_pb_ids_for_instance_ids(ins_seg)
        obj_id = self.task_obj_list[object_name].get_body_ids()
        print('obj_id in id_ins_seg: ', obj_id in id_ins_seg)
        if obj_id in id_ins_seg:
            yield from self._navigate_to(object_name, yaw=yaw)
            yield from self._place(object_name)
            print('Vis_place {}'.format(object_name))
        else:
            yield self._get_still_action()
            print('Vis_place {} ignored'.format(object_name))

