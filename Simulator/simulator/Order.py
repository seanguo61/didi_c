import numpy as np


class Order(object):
    __slots__ = ('_begin_p', '_end_p', '_begin_t',
                 '_t', '_price', '_waiting_time', '_assigned_time')

    def __init__(self, begin_position, end_position, begin_time, duration, price, wait_time):
        self._begin_p = begin_position  # node
        self._end_p = end_position      # node
        self._begin_t = begin_time
        # self._end_t = end_time
        self._t = duration              # the duration of order.
        self._price = price
        self._waiting_time = wait_time  # a order can last for "wait_time" to be taken
        self._assigned_time = -1

    def get_begin_position(self):
        return self._begin_p

    def get_begin_position_id(self):
        return self._begin_p.get_node_index()

    def get_end_position(self):
        return self._end_p

    def get_begin_time(self):
        return self._begin_t

    def set_assigned_time(self, city_time):
        self._assigned_time = city_time

    def get_assigned_time(self):
        return self._assigned_time

        # def get_end_time(self):
        #     return self._end_t

    def get_duration(self):
        return self._t

    def get_price(self):
        return self._price

    def get_wait_time(self):
        return self._waiting_time

