import numpy as np
from Simulator.simulator.utilities import *
from Simulator.simulator.Driver import *
from Simulator.simulator.Order import *


class Node(object):

    __slots__ = ('neighbors', '_index', 'orders', 'drivers',
                 'order_num', 'idle_driver_num', 'offline_driver_num',
                 'order_generator', 'n_side', 'layers_neighbors', 'layers_neighbors_id')

    def __init__(self, index):
        self._index = index
        self.neighbors = []
        self.orders = []
        self.drivers = {}
        self.order_num = 0
        self.idle_driver_num = 0  # number of idle drivers in this node
        self.offline_driver_num = 0
        self.order_generator = None

        self.n_side = 0      # the topology is a n-sided map
        self.layers_neighbors = []  # layer 1 indices: layers_neighbors[0] = [[1,1], [0, 1], ...],
        # layer 2 indices layers_neighbors[1]
        self.layers_neighbors_id = []  # layer 1: layers_neighbors_id[0] = [2, 1,.]

    def clean_node(self):
        self.orders = []
        self.order_num = 0
        self.drivers = {}
        self.idle_driver_num = 0
        self.offline_driver_num = 0

    def get_layers_neighbors(self, l_max, M, N, env):

        x, y = ids_1dto2d(self.get_node_index(), M, N)
        self.layers_neighbors = get_layers_neighbors(x, y, l_max, M, N)
        for layer_neighbors in self.layers_neighbors:
            temp = []
            for item in layer_neighbors:
                x, y = item
                node_id = ids_2dto1d(x, y, M, N)
                if env.nodes[node_id] is not None:
                    temp.append(node_id)
            self.layers_neighbors_id.append(temp)

    def get_node_index(self):
        return self._index

    def set_neighbors(self, nodes_list):
        self.neighbors = nodes_list
        self.n_side = len(nodes_list)
    # ******************司机相关*****************#
    def remove_driver(self, driver_id):

        removed_driver = self.drivers.pop(driver_id, None)
        self.idle_driver_num -= 1
        if removed_driver is None:
            raise ValueError('Nodes.remove_driver: Remove a driver that is not in this node')

        return removed_driver

    def add_driver(self, driver_id, driver):
        self.drivers[driver_id] = driver
        self.idle_driver_num += 1

    def get_driver_numbers(self):
        return self.idle_driver_num

    def get_idle_driver_numbers_loop(self):
        temp_idle_driver = 0
        for key, driver in self.drivers.items():
            if driver.onservice is False and driver.online is True:
                temp_idle_driver += 1
        return temp_idle_driver

    def get_off_driver_numbers_loop(self):
        temp_idle_driver = 0
        for key, driver in self.drivers.items():
            if driver.onservice is False and driver.online is False:
                temp_idle_driver += 1
        return temp_idle_driver

    # ******************订单相关*****************#

    def add_order_real(self, city_time, destination_node, duration, price):

        current_node_id = self.get_node_index()
        self.orders.append(Order(self,
                                 destination_node,
                                 city_time,
                                 duration,
                                 price, 0))
        self.order_num += 1

    def order_grouping(self, order_by_node):  # 针对多订单
        order_by_node = []
    # ******************司机状态相关**************#

    def remove_idle_driver_random(self):
        """Randomly remove one idle driver from current grid"""
        removed_driver_id = "NA"
        for key, item in self.drivers.items():
            if item.onservice is False and item.online is True:
                self.remove_driver(key)
                removed_driver_id = key
            if removed_driver_id != "NA":
                break
        assert removed_driver_id != "NA"
        return removed_driver_id

    def set_idle_driver_offline_random(self):
        """Randomly set one idle driver offline"""
        # 随机使一个司机下线，状态设置为offline
        removed_driver_id = "NA"
        for key, item in self.drivers.items():
            if item.onservice is False and item.online is True:
                item.set_offline()
                removed_driver_id = key
            if removed_driver_id != "NA":
                break
        assert removed_driver_id != "NA"
        return removed_driver_id

    def set_offline_driver_online(self):

        online_driver_id = "NA"
        for key, item in self.drivers.items():
            if item.onservice is False and item.online is False:
                item.set_online()
                online_driver_id = key
            if online_driver_id != "NA":
                break
        assert online_driver_id != "NA"
        return online_driver_id

    # ***************订单分配相关************************ #

    def remove_unfinished_order(self, city_time):
        # 移除未完成的订单（1.未被分派，2.order completed）
        # 即订单 开始时间 加上 等待时间（每个订单定义了一个可以等待的时间界限） 小于城市时间
        un_finished_order_index = []
        for idx, o in enumerate(self.orders):
            # order un served
            if o.get_wait_time()+o.get_begin_time() < city_time:
                un_finished_order_index.append(idx)

            # order completed
            if o.get_assigned_time() + o.get_duration() == city_time and o.get_assigned_time() != -1:
                un_finished_order_index.append(idx)

        if len(un_finished_order_index) != 0:
            # remove unfinished orders
            self.orders = [i for j, i in enumerate(self.orders) if j not in un_finished_order_index]
            self.order_num = len(self.orders)

    def simple_order_assign_real(self, city_time, city):

        reward = 0
        num_assigned_order = min(self.order_num, self.idle_driver_num)
        # 节点node 内 要分派的订单数等于 最小值（node中的订单数目，空闲司机数目）

        served_order_index = []
        for idx in np.arange(num_assigned_order):
            order_to_serve = self.orders[idx]
            order_to_serve.set_assigned_time(city_time)
            self.order_num -= 1
            reward += order_to_serve.get_price()
            served_order_index.append(idx)

            for key, assigned_driver in self.drivers.items():
                if assigned_driver.onservice is False and assigned_driver.online is True:

                    if order_to_serve.get_end_position() is not None:
                        assigned_driver.take_order(order_to_serve)
                        removed_driver = self.drivers.pop(assigned_driver.get_driver_id(), None)
                        assert removed_driver is not None
                    else:
                        assigned_driver.set_offline()  # order destination is not in target region 订单目的地不在目标区域
                    city.n_drivers -= 1
                    break

        all_order_num = len(self.orders)
        finished_order_num = len(served_order_index)

        # remove served orders
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_order_index]
        assert self.order_num == len(self.orders)
        # 返回的reward是该节点内全部订单的价格 加和
        return reward, all_order_num, finished_order_num

    def simple_order_assign_broadcast_update(self, city, neighbor_node_reward):

        assert self.idle_driver_num == 0
        reward = 0
        num_finished_orders = 0
        for neighbor_node in self.neighbors:
            if neighbor_node is not None and neighbor_node.idle_driver_num > 0:
                num_assigned_order = min(self.order_num, neighbor_node.idle_driver_num)
                rr = self.utility_assign_orders_neighbor(city, neighbor_node, num_assigned_order)

                reward += rr
                neighbor_node_reward[neighbor_node.get_node_index()] += rr
                num_finished_orders += num_assigned_order
            if self.order_num == 0:
                break

        assert self.order_num == len(self.orders)
        return reward, num_finished_orders

    def utility_assign_orders_neighbor(self, city, neighbor_node, num_assigned_order):

        served_order_index = []
        reward = 0
        curr_city_time = city.city_time
        for idx in np.arange(num_assigned_order):
            order_to_serve = self.orders[idx]
            order_to_serve.set_assigned_time(curr_city_time)
            self.order_num -= 1
            reward += order_to_serve.get_price()
            served_order_index.append(idx)
            for key, assigned_driver in neighbor_node.drivers.items():
                if assigned_driver.onservice is False and assigned_driver.online is True:
                    if order_to_serve.get_end_position() is not None:
                        assigned_driver.take_order(order_to_serve)
                        removed_driver = neighbor_node.drivers.pop(assigned_driver.get_driver_id(), None)
                        assert removed_driver is not None
                    else:
                        assigned_driver.set_offline()
                    city.n_drivers -= 1
                    break
        # remove served orders
        self.orders = [i for j, i in enumerate(self.orders) if j not in served_order_index]
        assert self.order_num == len(self.orders)
        return reward


