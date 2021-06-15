import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, JR2_Kinova, Fetch
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import EmptyScene
from gibson2.core.physics.interactive_objects import InteractiveObj, BoxShape, YCBObject
from gibson2.utils.utils import parse_config
from gibson2.core.render.profiler import Profiler

import pytest
import pybullet as p
import numpy as np
from gibson2.external.pybullet_tools.utils import set_base_values, joint_from_name, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, user_input, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, \
    set_point, create_box, stable_z, control_joints, get_max_limits, get_min_limits, get_sample_fn

import time
import numpy as np

config = parse_config('../configs/jr_interactive_nav.yaml')
s = Simulator(mode='gui', timestep=1 / 240.0, resolution=2048)
scene = EmptyScene()
s.import_scene(scene)
fetch = Fetch(config)
s.import_robot(fetch)

robot_id = fetch.robot_ids[0]

#arm_joints = joints_from_names(robot_id, ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_flex_joint', 'wrist_flex_joint'])
arm_joints = joints_from_names(robot_id, ['torso_lift_joint','shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint',
                                          'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint'])
#finger_joints = joints_from_names(robot_id, ['l_gripper_finger_joint', 'r_gripper_finger_joint'])
fetch.robot_body.reset_position([0, 0, 0])
fetch.robot_body.reset_orientation([0, 0, 1, 0])
x,y,z = fetch.get_end_effector_position()
#set_joint_positions(robot_id, finger_joints, [0.04,0.04])

print(x,y,z)
visual_marker = p.createVisualShape(p.GEOM_SPHERE, radius = 0.02)
marker = p.createMultiBody(baseVisualShapeIndex = visual_marker)

max_limits = [0,0] + get_max_limits(robot_id, arm_joints) + [0.05,0.05]
min_limits = [0,0] + get_min_limits(robot_id, arm_joints) + [0,0]
rest_position = [0,0] + list(get_joint_positions(robot_id, arm_joints)) + [0.04,0.04]

rest_position = (0,0,0.30322468280792236, -1.414019864768982,
                         1.5178184935241699, 0.8189625336474915,
                         2.200358942909668, 2.9631312579803466,
                         -1.2862852996643066, 0.0008453550418615341)

joint_range = list(np.array(max_limits) - np.array(min_limits))
joint_range = [item + 1 for item in joint_range]
jd = [0.1 for item in joint_range]
print(max_limits)
print(min_limits)


def accurateCalculateInverseKinematics(robotid, endEffectorId, targetPos, threshold, maxIter):

  sample_fn = get_sample_fn(robotid, arm_joints)
  set_joint_positions(robotid, arm_joints, sample_fn())
  closeEnough = False
  iter = 0
  dist2 = 1e30
  while (not closeEnough and iter < maxIter):
    #jointPoses = rest_position
    jointPoses = p.calculateInverseKinematics(robotid, endEffectorId, targetPos,
                                                     lowerLimits = min_limits,
                                                     upperLimits = max_limits,
                                                     jointRanges = joint_range,
                                                     restPoses = rest_position,
                                                     jointDamping = jd)
    set_joint_positions(robotid, arm_joints, jointPoses[2:10])
    ls = p.getLinkState(robotid, endEffectorId)
    newPos = ls[4]
    diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
    dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
    closeEnough = (dist2 < threshold)
    iter = iter + 1
  print ("Num iter: "+str(iter) + " threshold: "+str(dist2))
  return jointPoses

while True:
    with Profiler("Simulation: step"):
        fetch.robot_body.reset_position([0, 0, 0])
        fetch.robot_body.reset_orientation([0, 0, 1, 0])
        threshold = 0.001
        maxIter = 100
        joint_pos = accurateCalculateInverseKinematics(robot_id, fetch.parts['gripper_link'].body_part_index, [x, y, z], threshold, maxIter)[2:10]

        #set_joint_positions(robot_id, finger_joints, [0.04, 0.04])
        s.step()
        keys = p.getKeyboardEvents()
        for k, v in keys.items():

            if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN)):
                x += 0.01
            if (k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN)):
                x -= 0.01
            if (k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN)):
                y += 0.01
            if (k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN)):
                y -= 0.01
            if (k == ord('z') and (v & p.KEY_IS_DOWN)):
                z += 0.01
            if (k == ord('x') and (v & p.KEY_IS_DOWN)):
                z -= 0.01

        p.resetBasePositionAndOrientation(marker, [x,y,z], [0,0,0,1])

s.disconnect()