# Credits
# This file is adapted from Open Source projects. You can find the source code of their open source projects
# below. We acknowledgde and are gradteful to these contributors for their contributions.
# 
# Caelan Reed Garrett. PyBullet Planning. https://pypi.org/project/pybullet-planning/. 2020.

import math
import os
import random
import re
from collections import namedtuple
from itertools import combinations

import numpy as np

from .pr2_never_collisions import NEVER_COLLISIONS
from .utils import get_link_pose, set_joint_position, set_joint_positions, get_joint_positions, get_min_limit, get_max_limit,\
    link_from_name, Pose, joints_from_names, get_body_name, get_num_joints, Euler, get_links, get_link_name, PI

# TODO: restrict number of pr2 rotations to prevent from wrapping too many times

LEFT_ARM = 'left'
RIGHT_ARM = 'right'
ARM_NAMES = (LEFT_ARM, RIGHT_ARM)

def side_from_arm(arm):
    side = arm.split('_')[0]
    assert side in ARM_NAMES
    return side

side_from_gripper = side_from_arm

def arm_from_arm(arm): # TODO: deprecate
    side = side_from_arm(arm)
    assert (side in ARM_NAMES)
    return '{}_arm'.format(side)

arm_from_side = arm_from_arm

def gripper_from_arm(arm): # TODO: deprecate
    side = side_from_arm(arm)
    assert (side in ARM_NAMES)
    return '{}_gripper'.format(side)

gripper_from_side = gripper_from_arm

#####################################

PR2_GROUPS = {
    'base': ['x', 'y', 'theta'],
    'torso': ['torso_lift_joint'],
    'head': ['head_pan_joint', 'head_tilt_joint'],
    arm_from_arm(LEFT_ARM): ['l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_upper_arm_roll_joint',
                             'l_elbow_flex_joint', 'l_forearm_roll_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint'],
    arm_from_arm(RIGHT_ARM): ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint', 
                              'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint'],
    gripper_from_arm(LEFT_ARM): ['l_gripper_l_finger_joint', 'l_gripper_r_finger_joint',
                                 'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_tip_joint'],
    gripper_from_arm(RIGHT_ARM): ['r_gripper_l_finger_joint', 'r_gripper_r_finger_joint',
                                  'r_gripper_l_finger_tip_joint', 'r_gripper_r_finger_tip_joint'],
    # r_gripper_joint & l_gripper_joint are not mimicked
}

HEAD_LINK_NAME = 'high_def_optical_frame' # high_def_optical_frame | high_def_frame | wide_stereo_l_stereo_camera_frame
# kinect - 'head_mount_kinect_rgb_optical_frame' | 'head_mount_kinect_rgb_link'

PR2_TOOL_FRAMES = {
    LEFT_ARM: 'l_gripper_tool_frame',  # l_gripper_palm_link | l_gripper_tool_frame
    RIGHT_ARM: 'r_gripper_tool_frame',  # r_gripper_palm_link | r_gripper_tool_frame
    'head': HEAD_LINK_NAME,
}

# TODO: deprecate to use the parent of the gripper joints
PR2_GRIPPER_ROOTS = {
    LEFT_ARM: 'l_gripper_palm_link',
    RIGHT_ARM: 'r_gripper_palm_link',
}

PR2_BASE_LINK = 'base_footprint'

# Arm tool poses
#TOOL_POSE = ([0.18, 0., 0.], [0., 0.70710678, 0., 0.70710678]) # l_gripper_palm_link
TOOL_POSE = Pose(euler=Euler(pitch=np.pi/2)) # l_gripper_tool_frame (+x out of gripper arm)
#TOOL_DIRECTION = [0., 0., 1.]

#####################################

# Special configurations

TOP_HOLDING_LEFT_ARM = [0.67717021, -0.34313199, 1.2, -1.46688405, 1.24223229, -1.95442826, 2.22254125]
SIDE_HOLDING_LEFT_ARM = [0.39277395, 0.33330058, 0., -1.52238431, 2.72170996, -1.21946936, -2.98914779]
REST_LEFT_ARM = [2.13539289, 1.29629967, 3.74999698, -0.15000005, 10000., -0.10000004, 10000.]
WIDE_LEFT_ARM = [1.5806603449288885, -0.14239066980481405, 1.4484623937179126, -1.4851759349218694, 1.3911839347271555,
                 -1.6531320011389408, -2.978586584568441]
CENTER_LEFT_ARM = [-0.07133691252641006, -0.052973836083405494, 1.5741805775919033, -1.4481146328076862,
                   1.571782540186805, -1.4891468812835686, -9.413338322697955]
STRAIGHT_LEFT_ARM = np.zeros(7)
COMPACT_LEFT_ARM = [PI/4, 0., PI/2, -5*PI/8, PI/2, -PI/2, 5*PI/8] # TODO: generate programmatically

#COMPACT_LEFT_ARM = [PI/4, 0., PI/2, -5*PI/8, -PI/2, -PI/2, 3*PI/8] # More inward
#COMPACT_LEFT_ARM = [1*PI/8, 0., PI/2, -4*PI/8, -PI/2, -PI/2, 3*PI/8-PI/2] # Most inward

CLEAR_LEFT_ARM = [PI/2, 0., PI/2, -PI/2, PI/2, -PI/2, 0.]
# WIDE_RIGHT_ARM = [-1.3175723551150083, -0.09536552225976803, -1.396727055561703, -1.4433371993320296,
#                   -1.5334243909312468, -1.7298129320065025, 6.230244924007009]

PR2_LEFT_CARRY_CONFS = {
    'top': TOP_HOLDING_LEFT_ARM,
    'side': SIDE_HOLDING_LEFT_ARM,
}

#####################################

PR2_URDF = "models/pr2_description/pr2.urdf" # 87 joints
#PR2_URDF = "models/pr2_description/pr2_hpn.urdf"
#PR2_URDF = "models/pr2_description/pr2_kinect.urdf"
DRAKE_PR2_URDF = "models/drake/pr2_description/urdf/pr2_simplified.urdf" # 82 joints

def is_drake_pr2(robot): # 87
    return (get_body_name(robot) == 'pr2') and (get_num_joints(robot) == 82)

#####################################

# TODO: for when the PR2 is copied and loses it's joint names
# PR2_JOINT_NAMES = []
#
# def set_pr2_joint_names(pr2):
#     for joint in get_joints(pr2):
#         PR2_JOINT_NAMES.append(joint)
#
# def get_pr2_joints(joint_names):
#     joint_from_name = dict(zip(PR2_JOINT_NAMES, range(len(PR2_JOINT_NAMES))))
#     return [joint_from_name[name] for name in joint_names]

#####################################

def get_base_pose(pr2):
    return get_link_pose(pr2, link_from_name(pr2, PR2_BASE_LINK))

def rightarm_from_leftarm(config):
    right_from_left = np.array([-1, 1, -1, 1, -1, 1, -1])
    return config * right_from_left

def arm_conf(arm, left_config):
    side = side_from_arm(arm)
    if side == LEFT_ARM:
        return left_config
    elif side == RIGHT_ARM:
        return rightarm_from_leftarm(left_config)
    raise ValueError(side)

def get_carry_conf(arm, grasp_type):
    return arm_conf(arm, PR2_LEFT_CARRY_CONFS[grasp_type])

def get_other_arm(arm):
    for other_arm in ARM_NAMES:
        if other_arm != arm:
            return other_arm
    raise ValueError(arm)

#####################################

def get_disabled_collisions(pr2):
    #disabled_names = PR2_ADJACENT_LINKS
    #disabled_names = PR2_DISABLED_COLLISIONS
    disabled_names = NEVER_COLLISIONS
    #disabled_names = PR2_DISABLED_COLLISIONS + NEVER_COLLISIONS
    link_mapping = {get_link_name(pr2, link): link for link in get_links(pr2)}
    return {(link_mapping[name1], link_mapping[name2])
            for name1, name2 in disabled_names if (name1 in link_mapping) and (name2 in link_mapping)}


def load_dae_collisions():
    # pr2-beta-static.dae: link 0 = base_footprint
    # pybullet: link -1 = base_footprint
    dae_file = 'models/pr2_description/pr2-beta-static.dae'
    dae_string = open(dae_file).read()
    link_regex = r'<\s*link\s+sid="(\w+)"\s+name="(\w+)"\s*>'
    link_mapping = dict(re.findall(link_regex, dae_string))
    ignore_regex = r'<\s*ignore_link_pair\s+link0="kmodel1/(\w+)"\s+link1="kmodel1/(\w+)"\s*/>'
    disabled_collisions = []
    for link1, link2 in re.findall(ignore_regex, dae_string):
        disabled_collisions.append((link_mapping[link1], link_mapping[link2]))
    return disabled_collisions


def load_srdf_collisions():
    srdf_file = 'models/pr2_description/pr2.srdf'
    srdf_string = open(srdf_file).read()
    regex = r'<\s*disable_collisions\s+link1="(\w+)"\s+link2="(\w+)"\s+reason="(\w+)"\s*/>'
    disabled_collisions = []
    for link1, link2, reason in re.findall(regex, srdf_string):
        if reason == 'Never':
            disabled_collisions.append((link1, link2))
    return disabled_collisions

#####################################

def get_groups():
    return sorted(PR2_GROUPS)

def get_group_joints(robot, group):
    return joints_from_names(robot, PR2_GROUPS[group])

def get_group_conf(robot, group):
    return get_joint_positions(robot, get_group_joints(robot, group))

#get_group_position = get_group_conf

def set_group_conf(robot, group, positions):
    set_joint_positions(robot, get_group_joints(robot, group), positions)

def set_group_positions(robot, group_positions):
    for group, positions in group_positions.items():
        set_group_conf(robot, group, positions)

def get_group_positions(robot):
    return {group: get_group_conf(robot, group) for group in get_groups()}

#get_group_confs = get_group_positions

#####################################

# End-effectors

def get_arm_joints(robot, arm):
    return get_group_joints(robot, arm_from_arm(arm))


def get_torso_arm_joints(robot, arm):
    return joints_from_names(robot, PR2_GROUPS['torso'] + PR2_GROUPS[arm_from_arm(arm)])


#def get_arm_conf(robot, arm):
#    return get_joint_positions(robot, get_arm_joints(robot, arm))


def set_arm_conf(robot, arm, conf):
    set_joint_positions(robot, get_arm_joints(robot, arm), conf)


def get_gripper_link(robot, arm):
    assert arm in ARM_NAMES
    return link_from_name(robot, PR2_TOOL_FRAMES[arm])


# def get_gripper_pose(robot):
#    # world_from_gripper * gripper_from_tool * tool_from_object = world_from_object
#    pose = multiply(get_link_pose(robot, link_from_name(robot, LEFT_ARM_LINK)), TOOL_POSE)
#    #pose = get_link_pose(robot, link_from_name(robot, LEFT_TOOL_NAME))
#    return pose


def get_gripper_joints(robot, arm):
    return get_group_joints(robot, gripper_from_arm(arm))


def set_gripper_position(robot, arm, position):
    gripper_joints = get_gripper_joints(robot, arm)
    set_joint_positions(robot, gripper_joints, [position] * len(gripper_joints))


def open_arm(robot, arm): # These are mirrored on the pr2
    for joint in get_gripper_joints(robot, arm):
        set_joint_position(robot, joint, get_max_limit(robot, joint))


def close_arm(robot, arm):
    for joint in get_gripper_joints(robot, arm):
        set_joint_position(robot, joint, get_min_limit(robot, joint))

# TODO: use these names
open_gripper = open_arm
close_gripper = close_arm

#####################################