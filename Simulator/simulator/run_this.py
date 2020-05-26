import numpy as np
import pickle

from Simulator.simulator.envs import *


def running_example():
    mapped_matrix_int = np.arange(1, 101, 1).reshape([10, 10])
    mapped_matrix_int[0][0] = -100
    M, N = mapped_matrix_int.shape
    n_side = 8
    l_max = 2
    num_valid_grid = 99  # 这里合法的grid数目
    idle_driver_location_mat = pickle.load(
        open("/Users/guobaoshen/ipython_notebook/idle_driver_location_mat_py2.pkl", 'rb'))

    idle_driver_dist_time = pickle.load(
        open("/Users/guobaoshen/ipython_notebook/idle_driver_dist_time.pkl_py2.pkl", 'rb'))

    order_real = pickle.load(open("/Users/guobaoshen/ipython_notebook/orders_for_py2.pkl", 'rb'))

    onoff_driver_location_mat = []

    env = CityReal(mapped_matrix_int, idle_driver_dist_time, idle_driver_location_mat,
                   l_max, M, N, n_side, 1.0/10, np.array(order_real), np.array(onoff_driver_location_mat))

    state = env.reset_clean(1, 1, 0)
    order_response_rates = []
    T = 0
    max_iter = 144
    GMV = 0

    while T < max_iter:

        dispatch_action = []

        state, reward, info = env.step(dispatch_action, generate_order=2)

        print("City time {}: Order response rate: {}".format(env.city_time-1, env.order_response_rate))
        order_response_rates.append(env.order_response_rate)

        count = 0
        num_count = 0
        for key, _driver in env.drivers.items():
            if _driver.online:
                count += 1
        for ii in env.target_node_ids:
            num_count += env.nodes[ii].order_num
        print("num of orders: {}".format(num_count))
        print("idle driver: {} == {}".format(np.sum(state[0]), np.sum(env.get_observation_driver_state()) ))
        print("total num of online drivers: {}".format(count))

        T += 1
        GMV += reward
    print(" total response rate one episode: {}".format(np.mean(order_response_rates)))
    print("total reward one episode: {}".format(GMV))


if __name__ == "__main__":
    running_example()