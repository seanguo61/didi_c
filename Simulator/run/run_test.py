import pickle
import sys
import time
import tensorflow as tf
import numpy as np
from Simulator.simulator.envs import *
sys.path.append("../")


def collision_action(action_tuple):
    count = 0
    action_set = set(())
    for item in action_tuple:
        if item[1] == -1:
            continue
        grid_id_key = str(item[0]) + "-" + str(item[1])
        action_set.add(grid_id_key)
        conflict_id_key = str(item[1]) + "-" + str(item[0])
        if conflict_id_key in action_set:
            count += 1
    return count


# load data
log_dir = "logs/"
mkdir_p(log_dir)
current_time = time.strftime("%Y%m%d_%H-%M")
mapped_matrix_int = np.arange(1, 101, 1).reshape([10, 10])
mapped_matrix_int[0][0] = -100
M, N = mapped_matrix_int.shape
order_num_dist = []
num_valid_grid = 99  # 这里合法的grid数目
idle_driver_dist_time = pickle.load(open("/Users/guobaoshen/ipython_notebook/idle_driver_dist_time.pkl_py2.pkl", 'rb'))
idle_driver_location_mat = pickle.load(open("/Users/guobaoshen/ipython_notebook/idle_driver_location_mat_py2.pkl", 'rb'))
order_real = pickle.load(open("/Users/guobaoshen/ipython_notebook/orders_for_py2.pkl", 'rb'))
n_side = 6
l_max = 2

onoff_driver_location_mat = []
for tt in np.arange(144):
    m = np.zeros([100, 2])
    m[:, 0] = np.random.normal(-0.02, 0.02, [100])
    m[:, 1] = np.random.normal(2, 0.2, [100])
    onoff_driver_location_mat.append(m)
print("finish load data")


################## Initialize env ###################################

n_side = 6
GAMMA = 0.9
l_max = 9
RATIO = 1

env = CityReal(mapped_matrix_int, idle_driver_dist_time, idle_driver_location_mat,
               l_max, M, N, n_side, 1.0 / 10, np.array(order_real), np.array(onoff_driver_location_mat))

temp = np.array(env.target_grids) + env.M * env.N
target_id_states = env.target_grids + temp.tolist()

for n_iter in np.arange(25):

    curr_state = env.reset_clean(generate_order=1, ratio=RATIO, city_time=0)

    for ii in np.arange(145):
        action_tuple, valid_action_prob_mat = []

