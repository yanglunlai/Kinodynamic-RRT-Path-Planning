import json
from pybullet_tools.parse_json import parse_robot, parse_body
from pybullet_tools.utils import set_joint_positions, \
    wait_if_gui, wait_for_duration, get_collision_fn
from pybullet_tools.pr2_utils import get_disabled_collisions
import pybullet as p

def load_env(env_file):
    # load robot and obstacles defined in a json file
    with open(env_file, 'r') as f:
        env_json = json.loads(f.read())
    robots = {robot['name']: parse_robot(robot) for robot in env_json['robots']}
    bodies = {body['name']: parse_body(body) for body in env_json['bodies']}
    return robots, bodies

def get_collision_fn_PR2(robot, joints, obstacles):
    # check robot collision with environment
    disabled_collisions = get_disabled_collisions(robot)
    return get_collision_fn(robot, joints, obstacles=obstacles, attachments=[], \
        self_collisions=True, disabled_collisions=disabled_collisions)

def execute_trajectory(robot, joints, path, sleep=None):
    # Move the robot according to a given path
    if path is None:
        print('Path is empty')
        return
    print('Executing trajectory')
    for bq in path:
        set_joint_positions(robot, joints, bq)
        if sleep is None:
            wait_if_gui('Continue?')
        else:
            wait_for_duration(sleep)
    print('Finished')

def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id


def draw_line(start, end, width, color):
    line_id = p.addUserDebugLine(start, end, color, width)
    return line_id