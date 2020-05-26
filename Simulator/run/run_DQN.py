import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle, sys
sys.path.append("../")
import time
import tensorflow as tf
import numpy as np
from Simulator.simulator.envs import *


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



################## Load data ###################################
log_dir = "logs/"
mkdir_p(log_dir)
current_time = time.strftime("%Y%m%d_%H-%M")
order_time_dist = []
order_price_dist = []
mapped_matrix_int = np.arange(1, 101, 1).reshape([10, 10])
mapped_matrix_int[0][0] = -100
M, N = mapped_matrix_int.shape
order_num_dist = []
num_valid_grid = 99  # 这里合法的grid数目

# **************real_order***********#
idle_driver_dist_time = pickle.load(open("/Users/guobaoshen/ipython_notebook/idle_driver_dist_time.pkl_py2.pkl", 'rb'))
idle_driver_location_mat = pickle.load(open("/Users/guobaoshen/ipython_notebook/idle_driver_location_mat_py2.pkl", 'rb'))
order_real = pickle.load(open("/Users/guobaoshen/ipython_notebook/orders_for_py2.pkl", 'rb'))

'''
for i in range(144):
    idle_driver_dist_time[i][0] = idle_driver_dist_time[i][0]/2
idle_driver_location_mat = np.trunc(idle_driver_location_mat*0.5)
'''
n_side = 8
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

env = CityReal(mapped_matrix_int, idle_driver_dist_time, idle_driver_location_mat,
               l_max, M, N, n_side, 1.0 / 10, np.array(order_real), np.array(onoff_driver_location_mat))

temp = np.array(env.target_grids) + env.M * env.N
target_id_states = env.target_grids + temp.tolist()

print("******************* Finish generating one day order **********************")

print("******************* Starting training Deep SARSA **********************")
from Simulator.algorithm.DQN import *

MAX_ITER = 50  # 10 iteration the Q-learning loss will converge.
is_plot_figure = False
city_time_start = 0
EP_LEN = 144
global_step = 0
city_time_end = city_time_start + EP_LEN
EPSILON = 0.9
gamma = 0.9
learning_rate = 1e-2

prev_epsiode_reward = 0
all_rewards = []
order_response_rates = []
value_table_sum = []
episode_rewards = []
episode_conflicts_drivers = []
order_response_rate_episode = []
episode_dispatched_drivers = []
T = 144
action_dim = 7
state_dim = env.n_valid_grids * 3 + T

# tf.reset_default_graph()
sess = tf.Session()
tf.set_random_seed(1)
q_estimator = Estimator(sess, action_dim,
                        state_dim,
                        env,
                        scope="q_estimator",
                        summaries_dir=log_dir)

target_estimator = Estimator(sess, action_dim, state_dim, env, scope="target_q")
sess.run(tf.global_variables_initializer())
estimator_copy = ModelParametersCopier(q_estimator, target_estimator)
replay = ReplayMemory(memory_size=1e+6, batch_size=3000)
stateprocessor = stateProcessor(target_id_states, env.target_grids, env.n_valid_grids)

saver = tf.train.Saver()
save_random_seed = []
N_ITER_RUNS = 25
temp_value = 10
RATIO = 1
EPSILON_start = 0.5
EPSILON_end = 0.1
epsilon_decay_steps=15
epsilons = np.linspace(EPSILON_start, EPSILON_end, epsilon_decay_steps)
for n_iter in np.arange(25):
    RANDOM_SEED = n_iter + MAX_ITER - temp_value
    env.reset_randomseed(RANDOM_SEED)
    save_random_seed.append(RANDOM_SEED)
    batch_s, batch_a, batch_r = [], [], []
    batch_reward_gmv = []
    epsiode_reward = 0
    num_dispatched_drivers = 0

    # reset env
    is_regenerate_order = 1
    curr_state = env.reset_clean(generate_order=is_regenerate_order, ratio=RATIO, city_time=city_time_start)
    info = env.step_pre_order_assign(curr_state)
    context = stateprocessor.compute_context(info)
    curr_s = stateprocessor.utility_conver_states(curr_state)
    normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
    s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

    # record rewards to update the value table
    episodes_immediate_rewards = []
    num_conflicts_drivers = []
    curr_num_actions = []
    epsilon = epsilons[n_iter] if n_iter < 15 else EPSILON_end
    for ii in np.arange(EP_LEN + 1):

        # INPUT: state,  OUTPUT: action
        qvalues, action_idx, action_idx_valid, action_neighbor_idx, \
        action_tuple, action_starting_gridids = q_estimator.action(s_grid, context, epsilon)
        # a0

        # ONE STEP: r0
        next_state, r, info = env.step(action_tuple, 2)

        # r0
        immediate_reward = stateprocessor.reward_wrapper(info, curr_s)

        # a0
        action_mat = stateprocessor.to_action_mat(action_neighbor_idx)

        # s0
        s_grid_train = stateprocessor.to_grid_state_for_training(s_grid, action_starting_gridids)

        # s1
        s_grid_next = stateprocessor.to_grid_next_states(s_grid_train, next_state, action_idx_valid, env.city_time)

        # Save transition to replay memory
        if ii != 0:
            # r1, c0
            r_grid = stateprocessor.to_grid_rewards(action_idx_valid_prev, immediate_reward)
            targets_batch = r_grid + gamma * target_estimator.predict(s_grid_next_prev)

            # s0, a0, r1
            replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid_next_prev)

        state_mat_prev = s_grid_train
        action_mat_prev = action_mat
        context_prev = context
        s_grid_next_prev = s_grid_next
        action_idx_valid_prev = action_idx_valid

        # c1
        context = stateprocessor.compute_context(info[1])
        # s1
        curr_s = stateprocessor.utility_conver_states(next_state)
        normalized_curr_s = stateprocessor.utility_normalize_states(curr_s)
        s_grid = stateprocessor.to_grid_states(normalized_curr_s, env.city_time)  # t0, s0

        # Sample a minibatch from the replay memory and update q network training method1
        if replay.curr_lens != 0:
            for _ in np.arange(20):
                fetched_batch = replay.sample()
                mini_s, mini_a, mini_target, mini_next_s = fetched_batch
                q_estimator.update(mini_s, mini_a, mini_target, learning_rate, global_step)
                global_step += 1

        # Perform gradient descent update
        # book keeping
        global_step += 1
        all_rewards.append(r)
        batch_reward_gmv.append(r)
        order_response_rates.append(env.order_response_rate)
        num_conflicts_drivers.append(collision_action(action_tuple))
        curr_num_action = np.sum([aa[2] for aa in action_tuple]) if len(action_tuple) != 0 else 0
        curr_num_actions.append(curr_num_action)

    episode_reward = np.sum(batch_reward_gmv[1:])
    episode_rewards.append(episode_reward)
    n_iter_order_response_rate = np.mean(order_response_rates[1:])
    order_response_rate_episode.append(n_iter_order_response_rate)
    episode_conflicts_drivers.append(np.sum(num_conflicts_drivers[:-1]))
    episode_dispatched_drivers.append(np.sum(curr_num_actions[:-1]))

    print("iteration {} ********* reward {} order{} conflicts {} drivers {}".format(n_iter, episode_reward,
                                                                                             order_response_rate_episode[-1],
                                                                                             episode_conflicts_drivers[-1],
                                                                                             episode_dispatched_drivers[-1]))

    pickle.dump([episode_rewards, order_response_rate_episode, save_random_seed, episode_conflicts_drivers,
                 episode_dispatched_drivers], open(log_dir + "results.pkl", "wb"))

    if n_iter == 24:
        break

    # # training method 2.
    # for _ in np.arange(4000):
    #     fetched_batch = replay.sample()
    #     mini_s, mini_a, mini_target, mini_next_s = fetched_batch
    #     q_estimator.update(mini_s, mini_a, mini_target, learning_rate, global_step)
    #     global_step += 1

    # update target Q network
    estimator_copy.make(sess)


    saver.save(sess, log_dir+"model.ckpt")


