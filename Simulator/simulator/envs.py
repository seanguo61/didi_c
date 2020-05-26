import numpy as np

from Simulator.simulator.utilities import *
from Simulator.simulator.Driver import *
from Simulator.simulator.Order import *
from Simulator.simulator.objects import Node

RANDOM_SEED = 0


class CityReal:
    """A real city is consists of M*N grids """

    def __init__(self, mapped_matrix_int, idle_driver_dist_time, idle_driver_location_mat,
                 l_max, M, N, n_side, probability=1.0 / 28, real_orders="", onoff_driver_location_mat="",
                 global_flag="global", time_interval=10):

        self.M = M  # row numbers
        self.N = N  # column numbers
        self.nodes = [Node(i) for i in range(M * N)]  # a list of nodes: node id start from 0
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        self.n_offline_drivers = 0  # total number of offline drivers.
        self.construct_map_simulation(M, N, n_side)
        self.city_time = 0
        # self.idle_driver_distribution = np.zeros((M, N))
        self.n_intervals = int(1440 / time_interval)
        self.n_nodes = self.M * self.N
        self.n_side = n_side
        self.order_response_rate = 0

        self.RANDOM_SEED = RANDOM_SEED

        self.l_max = l_max
        assert 1 <= l_max <= 9  # Ignore orders less than 10 minutes and larger than 1.5 hours

        self.target_grids = []  # 在construct_node_real()函数中有赋值
        #
        self.n_valid_grids = 0  # num of valid grid
        self.nodes = [None for _ in np.arange(self.M * self.N)]
        self.construct_node_real(mapped_matrix_int)
        self.mapped_matrix_int = mapped_matrix_int

        self.construct_map_real(n_side)
        # self.order_num_dist = order_num_dist
        self.distribution_name = "Poisson"
        self.idle_driver_dist_time = idle_driver_dist_time
        self.idle_driver_location_mat = idle_driver_location_mat

        #self.order_time_dist = order_time_dist[:l_max] / np.sum(order_time_dist[:l_max])
        #self.order_price_dist = order_price_dist

        target_node_ids = []
        target_grids_sorted = np.sort(mapped_matrix_int[np.where(mapped_matrix_int > 0)])
        # np.where(condition),输出满足条件condition的元素的下标，这句话就是保留矩阵中大于0的，然后排序
        for item in target_grids_sorted:
            x, y = np.where(mapped_matrix_int == item)
            target_node_ids.append(ids_2dto1d(x, y, M, N))
        self.target_node_ids = target_node_ids
        # store valid note id. Sort by number of orders emerged. descending.

        self.node_mapping = {}
        self.construct_mapping()

        self.real_orders = real_orders  # 4 weeks' data
        # [[92, 300, 143, 2, 13.2],...] origin grid, destination grid, start time, end time, price.

        self.p = probability  # sample probability
        self.time_keys = [int(dt.strftime('%H%M')) for dt in
                          datetime_range(datetime(2017, 9, 1, 0), datetime(2017, 9, 2, 0),
                                         timedelta(minutes=time_interval))]
        self.day_orders = []  # one day's order.

        self.onoff_driver_location_mat = onoff_driver_location_mat

        # Stats
        self.all_grids_on_number = 0  # current online # drivers.
        self.all_grids_off_number = 0

        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))
        self.global_flag = global_flag
        self.weights_layers_neighbors = [1.0, np.exp(-1), np.exp(-2)]

    # *************初始化地图*****************#

    def reset_randomseed(self, random_seed):
        self.RANDOM_SEED = random_seed

    def construct_map_simulation(self, M, N, n):  # 设置 节点的邻居列表：node list
        """Connect node to its neighbors based on a simulated M by N map
            :param M: M row index matrix
            :param N: N column index matrix
            :param n: n - sided polygon
        """
        for idx, current_node in enumerate(self.nodes):
            if current_node is not None:
                i, j = ids_1dto2d(idx, M, N)

                current_node.set_neighbors(get_neighbor_list(i, j, M, N, n, self.nodes))

    def construct_node_real(self, mapped_matrix_int):
        """ Initialize node, only valid node in mapped_matrix_int will be initialized.
        mapped_matrix_int 是一个二维矩阵。 M * N
        """
        row_inds, col_inds = np.where(mapped_matrix_int >= 0)
        '''
        np.where()返回一个数组，分别是满足条件的节点的行和列坐标，
        row_inds, col_inds记录下了行坐标数组和列坐标数组

        target_ids=[]是：由合法的node_id组成的list，其中node_id是已经转化成1d后的grid id
        target_grids = target_ids
        '''
        target_ids = []  # start from 0.
        for x, y in zip(row_inds, col_inds):  # 将行列坐标打包成元祖，即（ ）
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id] = Node(node_id)  # node id start from 0.
            target_ids.append(node_id)

        for x, y in zip(row_inds, col_inds):
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id].get_layers_neighbors(self.l_max, self.M, self.N, self)

        self.target_grids = target_ids
        self.n_valid_grids = len(target_ids)

    def construct_map_real(self, n_side):
        """Build node connection.
        """
        for idx, current_node in enumerate(self.nodes):
            i, j = ids_1dto2d(idx, self.M, self.N)
            if current_node is not None:
                current_node.set_neighbors(get_neighbor_list(i, j, self.M, self.N, n_side, self.nodes))

    def construct_mapping(self):
        """
        :return:
        """
        target_grid_id = self.mapped_matrix_int[np.where(self.mapped_matrix_int > 0)]
        # 把非零的（合法）的格子grid id 取出来

        for g_id, n_id in zip(target_grid_id, self.target_grids):
            self.node_mapping[g_id] = n_id

    # *************订单相关*****************#

    def utility_bootstrap_oneday_order(self):
        # 从全部订单数据中sample出一天的数据
        num_all_orders = len(self.real_orders)
        index_sampled_orders = np.where(np.random.binomial(1, self.p, num_all_orders) == 1)
        one_day_orders = self.real_orders[index_sampled_orders]

        self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))

        day_orders = [[] for _ in np.arange(self.n_intervals)]
        for iorder in one_day_orders:
            #  iorder: [92, 300, 143, 2, 13.2]
            start_time = int(iorder[2])
            if iorder[0] not in self.node_mapping.keys() and iorder[1] not in self.node_mapping.keys():
                continue
            start_node = self.node_mapping.get(iorder[0], -100)
            end_node = self.node_mapping.get(iorder[1], -100)
            duration = int(iorder[3])
            price = iorder[4]

            if start_node == -100:
                column_index = self.target_grids.index(end_node)
                self.out_grid_in_orders[(start_time + duration) % self.n_intervals, column_index] += 1
                continue

            day_orders[start_time].append([start_node, end_node, start_time, duration, price])
        self.day_orders = day_orders

    def step_bootstrap_order_real(self, day_orders_t):
        # 从订单数据中取出每一时刻的订单数据，
        # 然后根据每个订单的起始节点，将订单 导入到相应节点node中，作为order对象
        for iorder in day_orders_t:
            start_node_id = iorder[0]
            end_node_id = iorder[1]
            start_node = self.nodes[start_node_id]

            if end_node_id in self.target_grids:
                end_node = self.nodes[end_node_id]
            else:
                end_node = None
            start_node.add_order_real(self.city_time, end_node, iorder[3], iorder[4])
            # add_order_real：参数： 时间，目的地，duraton， price

    # *************司机及司机状态相关*****************#

    def utility_get_n_idle_drivers_real(self):
        """
        control the number of idle drivers in simulator;
        idle_driver_dist_time,该list中是某时刻全部（遍历）节点内司机数目的均值和方差
        :return:
        """
        time = self.city_time % self.n_intervals
        mean, std = self.idle_driver_dist_time[time]
        np.random.seed(self.city_time)
        return np.round(np.random.normal(mean, std, 1)[0]).astype(int)

    def step_driver_status_control(self):
        # 对环境env中 的所有司机 的状态进行检查， 更新
        # Deal with orders finished at time T=1, check driver status. finish order, set back to off service
        for key, _driver in self.drivers.items():  # 是指对环境env中的所有司机 进行遍历
            _driver.status_control_eachtime(self)
            # status_control_eachtime是driver类的成员函数
            # 主要是对  调用该函数的driver状态 进行更新
        moment = self.city_time % self.n_intervals
        # 这里的moment也是指的是 city_time??,只不过取余了一下144
        orders_to_on_drivers = self.out_grid_in_orders[moment, :]
        # out_grid_in_orders：是env的一个成员变量，shape：144* len(self.target_grids))，初始化为全零矩阵
        # self.out_grid_in_orders = np.zeros((self.n_intervals, len(self.target_grids)))
        # target_grids： env中所有valid grid 的id 的集合，这里的id是将二维坐标转化为一位后的id，即2dto1d
        for idx, item in enumerate(orders_to_on_drivers):

            if item != 0:
                node_id = self.target_grids[idx]
                self.utility_add_driver_real_nodewise(node_id, int(item))

    def utility_add_driver_real_nodewise(self, node_id, num_added_driver):

        while num_added_driver > 0:
            if self.nodes[node_id].offline_driver_num > 0:
                self.nodes[node_id].set_offline_driver_online()
                self.n_drivers += 1
                self.n_offline_drivers -= 1
            else:
                # 在while循环中每次向node节点中新添加一个司机（idle driver），因为已经
                # 没有offline的司机来做online处理了
                n_total_drivers = len(self.drivers.keys())
                added_driver_id = n_total_drivers
                self.drivers[added_driver_id] = Driver(added_driver_id)
                self.drivers[added_driver_id].set_position(self.nodes[node_id])
                self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])
                self.n_drivers += 1
            num_added_driver -= 1

    def utility_set_drivers_offline_real_nodewise(self, node_id, n_drivers_to_off):
        # 在某一节点内下线相应数量的司机
        while n_drivers_to_off > 0:
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random()
                self.n_drivers -= 1
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1
                self.all_grids_off_number += 1
            else:
                break

    def step_driver_online_offline_control_new(self, n_idle_drivers):

        # 这个函数只在reset_clean中被调用过，在其他地方没有被调用过，调用时self.n_drivers = 0
        offline_drivers = self.utility_collect_offline_drivers_id()
        # 统计多少司机是offline，返回offline司机 的id的 list
        self.n_offline_drivers = len(offline_drivers)
        # offline状态的司机数目
        # self.n_drivers = 0  # total idle number of drivers. online and not on service.
        if n_idle_drivers > self.n_drivers:  # 需要往环境中增加在线online状态的司机

            self.utility_add_driver_real_new_offlinefirst(n_idle_drivers - self.n_drivers)

        elif n_idle_drivers < self.n_drivers:  # 需要使环境中的一些司机下线
            self.utility_set_drivers_offline_real_new(self.n_drivers - n_idle_drivers)
        else:
            pass

    def utility_collect_offline_drivers_id(self):
        """count how many drivers are offline
        :return: offline_drivers: a list of offline driver id
        """
        count = 0  # offline driver num
        offline_drivers = []  # record offline driver id
        for key, _driver in self.drivers.items():
            if _driver.online is False:
                count += 1
                offline_drivers.append(_driver.get_driver_id())
        return offline_drivers

    def utility_add_driver_real_new_offlinefirst(self, num_added_driver):

        # curr_idle_driver_distribution = self.get_observation()[0][np.where(self.mapped_matrix_int > 0)]
        curr_idle_driver_distribution = self.get_observation()[0]
        # 当前 空闲司机分布
        curr_idle_driver_distribution_resort = np.array(
            [int(curr_idle_driver_distribution.flatten()[index]) for index in
             self.target_node_ids])
        # 压平，将二维分布变成一个格子长度的一位数组分布
        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]
        # idle_driver_location_mat： 144 * num_of_grids ，每一项表示该格子内司机数的均值

        idle_diff = idle_driver_distribution.astype(int) - curr_idle_driver_distribution_resort
        # 用idle_driver_location_mat中提取出的数目 减去 当前观测值中统计得到空闲司机的数目
        idle_diff[np.where(idle_diff <= 0)] = 0

        if float(np.sum(idle_diff)) == 0:
            return
        np.random.seed(self.RANDOM_SEED)
        # np.random.choice 从 self.target_node_ids 按概率选择出给定个数个 节点（id）（节点id可以有重复，即被重复抽到）
        node_ids = np.random.choice(self.target_node_ids, size=[num_added_driver],
                                    p=idle_diff / float(np.sum(idle_diff)))

        for ii, node_id in enumerate(node_ids):

            if self.nodes[node_id].offline_driver_num > 0:
                self.nodes[node_id].set_offline_driver_online()
                self.n_drivers += 1  # 空闲司机 idle_drivers 的数目
                self.n_offline_drivers -= 1
            else:  # 如果该节点内已经不存在处于离线状态下offline的司机了
                # 就只能new一个driver对象， 添加一个司机
                n_total_drivers = len(self.drivers.keys())
                added_driver_id = n_total_drivers
                self.drivers[added_driver_id] = Driver(added_driver_id)
                self.drivers[added_driver_id].set_position(self.nodes[node_id])
                self.nodes[node_id].add_driver(added_driver_id, self.drivers[added_driver_id])
                self.n_drivers += 1

    def utility_set_drivers_offline_real_new(self, n_drivers_to_off):

        # 调整当前curr_idle_driver_distribution空闲司机的分布，设置一部分司机下线offline
        curr_idle_driver_distribution = self.get_observation()[0]
        curr_idle_driver_distribution_resort = np.array([int(curr_idle_driver_distribution.flatten()[index])
                                                         for index in self.target_node_ids])

        # historical idle driver distribution
        idle_driver_distribution = self.idle_driver_location_mat[self.city_time % self.n_intervals, :]

        # diff of curr idle driver distribution and history
        idle_diff = curr_idle_driver_distribution_resort - idle_driver_distribution.astype(int)
        idle_diff[np.where(idle_diff <= 0)] = 0

        n_drivers_can_be_off = int(np.sum(curr_idle_driver_distribution_resort[np.where(idle_diff >= 0)]))
        if n_drivers_to_off > n_drivers_can_be_off:
            n_drivers_to_off = n_drivers_can_be_off

        sum_idle_diff = np.sum(idle_diff)
        if sum_idle_diff == 0:
            return
        np.random.seed(self.RANDOM_SEED)
        node_ids = np.random.choice(self.target_node_ids, size=[n_drivers_to_off],
                                    p=idle_diff / float(sum_idle_diff))

        for ii, node_id in enumerate(node_ids):
            if self.nodes[node_id].idle_driver_num > 0:
                self.nodes[node_id].set_idle_driver_offline_random()
                self.n_drivers -= 1  # 下线一部分司机
                self.n_offline_drivers += 1
                n_drivers_to_off -= 1

    # *************订单分配相关*****************#
    def step_add_dispatched_drivers(self, save_remove_id):
        # drivers dispatched at t, arrived at t + 1
        for destination_node_id, arrive_driver_id in save_remove_id:
            self.drivers[arrive_driver_id].set_position(self.nodes[destination_node_id])
            self.drivers[arrive_driver_id].set_online_for_finish_dispatch()
            self.nodes[destination_node_id].add_driver(arrive_driver_id, self.drivers[arrive_driver_id])
            self.n_drivers += 1

    def step_dispatch_invalid(self, dispatch_actions):
        """ If a
        :param dispatch_actions:
        :start_node_id, end_node_id, num_of_drivers = action
        :return:
        :self.nodes = [Node(i) for i in xrange(M * N)]  # a list of nodes: node id start from 0
        """

        save_remove_id = []
        for action in dispatch_actions:
            # dispatch_action 是一个联合动作，
            start_node_id, end_node_id, num_of_drivers = action  # 起始id，目的地id，司机数目
            if self.nodes[start_node_id] is None or num_of_drivers == 0:  # 起始id为空或数目为0
                continue  # not a feasible action

            if self.nodes[start_node_id].get_driver_numbers() < num_of_drivers:
                # 如果起始格子中的司机数目小于调度的司机数目,调度数目等于起始格子中司机数目
                num_of_drivers = self.nodes[start_node_id].get_driver_numbers()

            if end_node_id < 0:  # 目的地id小于0，为不存在的区域，就将调度的司机下线
                for _ in np.arange(num_of_drivers):  # 同一格子中司机是同构的
                    self.nodes[start_node_id].set_idle_driver_offline_random()
                    # set_idle_driver_offline_random 随机让一个空闲司机下线，该函数为node类的成员函数
                    self.n_drivers -= 1
                    self.n_offline_drivers += 1
                    self.all_grids_off_number += 1
                continue

            if self.nodes[end_node_id] is None:  # 目的地id不存在，同样下线需要调度数量的司机
                for _ in np.arange(num_of_drivers):
                    self.nodes[start_node_id].set_idle_driver_offline_random()
                    self.n_drivers -= 1
                    self.n_offline_drivers += 1
                    self.all_grids_off_number += 1
                continue

            if self.nodes[end_node_id] not in self.nodes[start_node_id].neighbors:
                # 目的地不在起始格子的 邻居 范围内
                raise ValueError('City:step(): not a feasible dispatch')

            for _ in np.arange(num_of_drivers):
                # t = 1 dispatch start, idle driver decrease
                remove_driver_id = self.nodes[start_node_id].remove_idle_driver_random()
                # remove_idle_driver_random函数随机选择一个空闲司机移除，并返回其id
                save_remove_id.append((end_node_id, remove_driver_id))
                self.drivers[remove_driver_id].set_position(None)
                self.drivers[remove_driver_id].set_offline_for_start_dispatch()
                # 在起始id node内下线该司机
                self.n_drivers -= 1
                # save_remove_id是由(end_node_id, remove_driver_id)目的地id，移除司机id构成的一个列表

        return save_remove_id

    def step_assign_order_broadcast_neighbor_reward_update(self):
        """
        :该函数是env的成员函数
        :Consider the orders whose destination or origin is not in the target region
        :param num_layers:
        :param weights_layers_neighbors: [1, 0.5, 0.25, 0.125]
        :return:
        """
        node_reward = np.zeros((len(self.nodes)))
        neighbor_reward = np.zeros((len(self.nodes)))
        reward = 0  # R_{t+1}
        all_order_num = 0
        finished_order_num = 0
        for node in self.nodes:  # 遍历所有node
            if node is not None:  # 如果node非空
                reward_node, all_order_num_node, finished_order_num_node = node.simple_order_assign_real(self.city_time,
                                                                                                         self)
                reward += reward_node
                all_order_num += all_order_num_node
                finished_order_num += finished_order_num_node
                node_reward[node.get_node_index()] += reward_node  # 将该时刻该节点的reward放入数组中

        # Second round broadcast 第二轮分配，剩下的未分配的订单分配给其邻居节点（格子grid）
        for node in self.nodes:
            if node is not None:
                if node.order_num != 0:  # 如果node内的订单数还不为0，即还有未分配的订单
                    reward_node_broadcast, finished_order_num_node_broadcast \
                        = node.simple_order_assign_broadcast_update(self, neighbor_reward)
                    # 返回的reward_node_broadcast 是对某一个节点的邻居节点进行遍历后服务的全部订单的价值
                    reward += reward_node_broadcast
                    finished_order_num += finished_order_num_node_broadcast

        node_reward = node_reward + neighbor_reward
        if all_order_num != 0:
            self.order_response_rate = finished_order_num / float(all_order_num)
        else:
            self.order_response_rate = -1

        return reward, [node_reward, neighbor_reward]

    def step_pre_order_assign(self, next_state):

        remain_drivers = next_state[0] - next_state[1]
        remain_drivers[remain_drivers < 0] = 0

        remain_orders = next_state[1] - next_state[0]
        remain_orders[remain_orders < 0] = 0

        if np.sum(remain_orders) == 0 or np.sum(remain_drivers) == 0:
            context = np.array([remain_drivers, remain_orders])
            return context

        remain_orders_1d = remain_orders.flatten()
        remain_drivers_1d = remain_drivers.flatten()

        for node in self.nodes:  # 遍历node
            if node is not None:
                curr_node_id = node.get_node_index()
                if remain_orders_1d[curr_node_id] != 0:
                    for neighbor_node in node.neighbors:
                        if neighbor_node is not None:
                            neighbor_id = neighbor_node.get_node_index()
                            a = remain_orders_1d[curr_node_id]
                            b = remain_drivers_1d[neighbor_id]
                            remain_orders_1d[curr_node_id] = max(a - b, 0)
                            remain_drivers_1d[neighbor_id] = max(b - a, 0)
                        if remain_orders_1d[curr_node_id] == 0:
                            break

        context = np.array([remain_drivers_1d.reshape(self.M, self.N),
                            remain_orders_1d.reshape(self.M, self.N)])
        return context

    # *************环境的两个外部接口函数：step 和 reset_clean 函数*****************#
    def get_observation_driver_state(self):
        """ Get idle driver distribution, computing #drivers from node.
        :return:
        """
        next_state = np.zeros((self.M, self.N))
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[row_id, column_id] = _node.get_idle_driver_numbers_loop()

        return next_state

    def get_observation(self):
        next_state = np.zeros((2, self.M, self.N))
        # state的shape---[2,M,N]
        for _node in self.nodes:
            if _node is not None:
                #print(_node.get_node_index())
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                next_state[0, row_id, column_id] = _node.idle_driver_num
                next_state[1, row_id, column_id] = _node.order_num
                #print(row_id, column_id)
        return next_state

    def reset_clean(self, generate_order=1, ratio=1, city_time=""):
        if city_time != "":
            self.city_time = city_time

        # clean orders and drivers
        self.drivers = {}  # driver[driver_id] = driver_instance  , driver_id start from 0
        self.n_drivers = 0  # total idle number of drivers. online and not on service.
        self.n_offline_drivers = 0  # total number of offline drivers.
        for node in self.nodes:
            if node is not None:
                node.clean_node()

        # Generate one day's order.
        if generate_order == 1:
            self.utility_bootstrap_oneday_order()

        # Init orders of current time step
        moment = self.city_time % self.n_intervals
        self.step_bootstrap_order_real(self.day_orders[moment])

        # Init current driver distribution
        if self.global_flag == "global":
            num_idle_driver = self.utility_get_n_idle_drivers_real()
            num_idle_driver = int(num_idle_driver * ratio)
        else:
            num_idle_driver = self.utility_get_n_idle_drivers_nodewise()


        self.step_driver_online_offline_control_new(num_idle_driver)
        return self.get_observation()

    def step_increase_city_time(self):
        self.city_time += 1
        # set city time of drivers
        for driver_id, driver in self.drivers.items():
            driver.set_city_time(self.city_time)

    def step_remove_unfinished_orders(self):
        for node in self.nodes:
            if node is not None:
                node.remove_unfinished_order(self.city_time)

    def step(self, dispatch_actions, generate_order=1):
        info = []
        '''**************************** T = 1 ****************************'''
        # Loop over all dispatch action, change the driver distribution
        # action: (start_node_id, end_node_id, num_of_drivers)
        save_remove_id = self.step_dispatch_invalid(dispatch_actions)
        # When the drivers go to invalid grid, set them offline.
        # save_remove_id是由(end_node_id, remove_driver_id)目的地id，移除司机id 构成的一个列表
        # step_dispatch_invalid函数根据action，完成司机的分派调度工作，即司机位置更新

        reward, reward_node = self.step_assign_order_broadcast_neighbor_reward_update()

        '''**************************** T = 2 ****************************'''
        # increase city time t + 1
        self.step_increase_city_time()
        # 这里的self是env，即整个环境
        self.step_driver_status_control()  # drivers finish order become available again.

        # drivers dispatched at t, arrived at t + 1, become available at t+1
        self.step_add_dispatched_drivers(save_remove_id)

        # generate order at t + 1
        if generate_order == 1:
            self.step_generate_order_real()
        else:
            moment = self.city_time % self.n_intervals
            self.step_bootstrap_order_real(self.day_orders[moment])

        if self.global_flag == "global":
            num_idle_driver = self.utility_get_n_idle_drivers_real()
            # 当前时刻的idle 司机数
        self.step_driver_online_offline_control_new(num_idle_driver)

        self.step_remove_unfinished_orders()
        # get states S_{t+1}  [driver_dist, order_dist]
        next_state = self.get_observation()

        context = self.step_pre_order_assign(next_state)
        info = [reward_node, context]
        return next_state, reward, info




