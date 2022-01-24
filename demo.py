import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, wait_if_gui, load_pybullet


np.random.seed(1)
state_names = ('x', 'y', 'theta', 'u', 'v', 'w')
state_limits = {'x': (-4.75, 4.75), 'y': (-4.75, 4.75), 'theta': (-np.pi, np.pi), 'u': (-0.5, 0.5),
                'v': (-0.5, 0.5), 'w': (-0.3, 0.3)}

def dynamics(now_state, inputs):
    # state space(x,y,theta,u,v,w)
    m = 5  # mass
    I = 5  # inertia tensor
    dt = 0.15  # 0.075
    x = now_state[0]
    y = now_state[1]
    theta = now_state[2]
    u = now_state[3]
    v = now_state[4]
    w = now_state[5]
    fx = inputs[0]
    fy = inputs[1]
    ax = fx / m
    ay = fy / m
    alpha = inputs[2] / I
    next_u = u + ax * dt
    next_v = v + ay * dt
    next_w = w + alpha * dt
    u = next_u
    v = next_v
    w = next_w
    next_x = x + (u * np.cos(theta) - v * np.sin(theta)) * dt  # in the fixed frame
    next_y = y + (u * np.sin(theta) + v * np.cos(theta)) * dt  # in the fixed frame
    next_theta = theta + w * dt  # in the fixed frame
    while next_theta < -np.pi:
        next_theta = next_theta + 2 * np.pi
    while next_theta > np.pi:
        next_theta = next_theta - 2 * np.pi
    return (next_x, next_y, next_theta, next_u, next_v, next_w)


def taskspace(config):
    return (config[0], config[1], config[2])


def limit_check(config):
    for i in range(6):
        if (config[i] < state_limits[state_names[i]][0] or config[i] > state_limits[state_names[i]][1]) and i != 2:
            return False
    return True


def distance(now, next):
    distance_temp = 0
    weights = [4, 4, 3, 2, 2, 1]
    for i in range(6):
        if i != 2:
            distance_temp = distance_temp + (weights[i] * abs(next[i] - now[i]) ** 2)
        else:
            distance_temp = distance_temp + (weights[i] * (theta_diff(next[i], now[i])) ** 2)
    return np.sqrt(distance_temp)



# a and b are angles in radians within [-pi, pi], angle_diff returns the angle difference within [0 , pi][-pi, pi]
def theta_diff(t1, t2):
    diff = t1 - t2
    while diff < -np.pi:
        diff = diff + 2 * np.pi
    while diff > np.pi:
        diff = diff - 2 * np.pi
    return abs(diff)


def achieve(config1, config2):
    dx = abs(config2[0] - config1[0])
    dy = abs(config2[1] - config1[1])
    dtheta = theta_diff(config2[2], config1[2])
    du = abs(config2[3] - config1[3])
    dv = abs(config2[4] - config1[4])
    dw = abs(config2[5] - config1[5])
    if (np.sqrt(
            dx ** 2 + dy ** 2) < 0.125) and (dtheta < 0.15) and (np.sqrt(du ** 2 + dv ** 2) < 0.2) and dw < 0.2:
        return True
    return False


def get_rand(goal_config, goal_bias):
    prob = np.random.random(1)  # To make probability of picking the goal node instead of the random one.
    if prob <= goal_bias:
        return goal_config
    else:
        q_random = [0, 0, 0, 0, 0, 0]
        for i in range(len(state_limits)):
            temp_low_lim, temp_up_lim = state_limits[state_names[i]]
            q_random[i] = round(temp_low_lim + (temp_up_lim - temp_low_lim) * np.random.random(), 2)
        return (q_random[0], q_random[1], q_random[2], q_random[3], q_random[4], q_random[5])


def get_near(explored_nodes, q_random):
    d_list = []
    for node in explored_nodes:
        d_list.append(distance(node, q_random))
    min_ind = np.argmin(d_list)
    return explored_nodes[min_ind]


# TODO: Define primitives
def get_new(near, random, num):
    p1 = (1.0, 0.0, 0.0)  # Fx
    p2 = (0.0, 0.0, -1.0)  # rotate clockwise
    p3 = (0.0, 0.0, 1.0)  # rotate counterclockwise
    p4 = (-1.0, 0.0, 0.0)  # Fx
    p5 = (0.0, 1.0, 0.0)  # Fy
    p6 = (0.0, -1.0, 0.0)  # Fy
    p7 = (1.0, 1.0, 0.0)  # Fx and Fy
    p8 = (1.0, -1.0, 0.0)  # Fx and Fy
    p9 = (-1.0, 1.0, 0.0)  # Fx and Fy
    p10 = (-1.0, -1.0, 0.0)  # Fx and Fy
    p11 = (1.0, 0.0, 1.0)  # Fx and rotate
    p12 = (1.0, 0.0, -1.0)  # Fx and rotate
    p13 = (-1.0, 0.0, 1.0)  # Fx and rotate
    p14 = (-1.0, 0.0, -1.0)  # Fx and rotate
    p15 = (0.0, 1.0, 1.0)  # rotate and Fy
    p16 = (0.0, 1.0, -1.0)  # rotate and Fy
    p17 = (0.0, -1.0, 1.0)  # rotate and Fy
    p18 = (0.0, -1.0, -1.0)  # rotate and Fy
    primitive = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18]
    dist = []
    for i in range(num):
        dist.append(distance(dynamics(near, primitive[i]), random))
    min_id = np.argmin(dist)
    return dynamics(near, primitive[min_id])


def path_quality(ex, p):
    explored_node = len(ex)
    quality_1 = 0
    quality_2 = 0
    r = 0.3*np.sqrt(2)
    node_num = len(p)
    for i in range(node_num-1):
        dist_temp_2 = np.sqrt((p[i+1][0]-p[i][0])**2 + (p[i+1][1]-p[i][1])**2 + r * theta_diff(p[i+1][2], p[i][2]))
        dist_temp_1 = np.sqrt((p[i+1][0]-p[i][0])**2 + (p[i+1][1]-p[i][1])**2)
        quality_1 = quality_1 + dist_temp_1
        quality_2 = quality_2 + dist_temp_2
    return explored_node, node_num, np.sqrt(quality_1), np.sqrt(quality_2)


# RRT-Connect
def rrt_connect(start_config, goal_config, collision_fn, prim_num):
    path = []
    config_path = []
    #start_time = time.time()
    goal_bias = 0.1  # 10%
    root = -1
    explored = [start_config]
    parent = {}  # a dictionary key: tuple(a config), value: tuple(parent's config)
    parent[start_config] = root
    collision_times = 0
    finished = 0

    while finished == 0:
        random_finished = 0
        while random_finished == 0:
            random_config = get_rand(goal_config, goal_bias)
            if collision_fn(taskspace(random_config)) == False and limit_check(
                    random_config) == True:  # hit an obstacle
                random_finished = 1
        near_config = get_near(explored, random_config)

        connect_times = 0
        while connect_times < 120:  # to prevent the random sample is too far away
            # check if the near node hit obstacle or out of limit
            if collision_fn(taskspace(near_config)) == True or limit_check(near_config) == False:
                collision_times = collision_times + 1
                break  # get random node again

            if achieve(near_config, random_config):  # reach the random sample
                new_config = random_config
                if parent.get(new_config) is None:
                    parent[new_config] = near_config
                else:
                    break
                explored.append(new_config)
                break  # get random node again

            else:  # have not reached the random sample, then extend!
                # print("extend")
                new_config = get_new(near_config, random_config, prim_num)

            if collision_fn(taskspace(new_config)) == True or limit_check(new_config) == False:  # hit an obstacle
                collision_times = collision_times + 1
                break  # get random node again
            if parent.get(new_config) is None:
                parent[new_config] = near_config
            else:
                break
            explored.append(new_config)
            near_config = new_config  # the new node then become the nearest node to the random node.
            connect_times = connect_times + 1

            if achieve(near_config, goal_config):  # then, we check whether we arrive goal now
                current_config = near_config
                for i in range(len(parent)):
                    if parent[current_config] != root:
                        path.insert(0, taskspace(current_config))
                        config_path.insert(0, current_config)
                        current_config = parent[current_config]
                finished = 1
                break

    #computetime = time.time() - start_time
    #print("Planner run time(without drawing the points): ", computetime, "sec.")
    return explored, path, config_path, parent

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=False)
    # load robot and obstacle resources
    _, obstacles = load_env('environment.json')
    robot = load_pybullet("myrobot.urdf")
    # define active DoFs
    base_joints = [0, 1, 2]
    collision_fn = get_collision_fn_PR2(robot, base_joints, list(obstacles.values()))
    print()
    print()
    print("Building RRT...")
    print("We will show the search trees(blue) and executed path(green) in this program.")
    print("This program is expected to run for 4~10 minutes (incluing drawing paths)...")
    print()
    start_config = (-4.5, 4.5, 0, 0, 0, 0)
    goal_config = (4.5, -4.5, -np.pi/2, 0, 0, 0)
    #start_time = time.time()
    explored, path, config_path, dictionary = rrt_connect(start_config, goal_config, collision_fn, 18)
    totalnodes, nodenum, quality_eu, quality_plus = path_quality(explored, path)
    print("Finish building the RRT!!")
    #print("computation time:", time.time() - start_time)
    print("number of explored nodes:", totalnodes)
    print("number of path nodes:", nodenum)
    print("path quality(euclidean):", quality_eu)
    print("path quality(consider theta):", quality_plus)

    disconnect()
    connect(use_gui=True)
    _, obstacles = load_env('environment.json')
    robot = load_pybullet("myrobot.urdf")
    start_config = (-4.5, 4.5, 0)
    collision_fn(start_config)
    draw_sphere_marker((goal_config[0], goal_config[1], 0.1), 0.15, (1, 0, 0, 1))
    print("Start drawing the explored path (Blue)")
    for path_i in explored:
        draw_sphere_marker((path_i[0], path_i[1], 0.06), 0.05, (0, 0, 1, 1))
    print("Start drawing the path (green)")
    for path_i in path:
        draw_sphere_marker((path_i[0], path_i[1], 0.08), 0.05, (0, 1, 0, 1))

    ######################
    print("Start executing the path")
    # Execute planned path
    execute_trajectory(robot, base_joints, path, sleep=0.1)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()


if __name__ == '__main__':
    main()
