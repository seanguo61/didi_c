import numpy as np
from Simulator.simulator.Order import *


class order_group(object):
    __slots__ = ('_begin_p', '_end_p', '_begin_t',
                 '_t', '_price', '_waiting_time', '_assigned_time', 'num_order_group', 'order_list')

    def __init__(self):
        self.num_order_group = 0
        self.order_list = []
        self._price = 0
        self._begin_p = None
        self._end_p = None
        self._begin_t = None
        self._t = None
        self._waiting_time = None
        self._assigned_time = None

    def add_order(self, order):
        self.order_list.append(order)
        self.num_order_group += 1
        self._begin_t = order.get_begin_time()
        self._begin_p = order.get_begin_position()
        self._price += order.get_price()
        # 下面是需要更新的，包括目的地，总时长,当前先简化简化 再简化
        self._end_p = order.get_end_position()
        self._t = order.get_duration()
        self._waiting_time = order.get_wait_time()
        self._assigned_time = order.get_begin_position_id()

