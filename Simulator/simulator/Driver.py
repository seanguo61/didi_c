import numpy as np


class Driver(object):
    __slots__ = ("online", "onservice", 'node', 'city_time', 'order', '_driver_id',
                 'order_list')  # orders 表示聚类后 的订单分组

    def __init__(self, driver_id):
        self._driver_id = driver_id
        self.node = None
        self.order = None
        self.order_list = []  # 司机接受的订单分组， 一个list，需要append新订单
        self.online = True
        self.onservice = False
        self.city_time = 0


    def set_position(self, node):
        self.node = node

    def get_driver_id(self):
        return self._driver_id

    def update_city_time(self):
        self.city_time += 1

    def set_city_time(self, city_time):
        self.city_time = city_time

    def set_order_start(self, order):  # 需要修改
        self.order = order

    def set_order_finish(self):  # 需要修改
        self.order = None
        self.onservice = False

    def set_offline(self):
        assert self.onservice is False and self.online is True
        self.online = False
        self.node.idle_driver_num -= 1
        self.node.offline_driver_num += 1

    def set_offline_for_start_dispatch(self):
        assert self.onservice is False
        self.online = False

    def set_online(self):
        assert self.onservice is False
        self.online = True
        self.node.idle_driver_num += 1
        self.node.offline_driver_num -= 1

    def set_online_for_finish_dispatch(self):
        self.online = True
        assert self.onservice is False

    def take_order(self, order):
        """ take order, driver show up at destination when order is finished
            该函数的作用即司机driver接受订单
            在didi问题描述中，一个司机在同时只能匹配一个订单，
            因此driver类中的order变量只有一个
            在    饿了么问题  中，order的值是一个列表，即为一批订单（多个订单）
            即order[]=[order1, order2, order3, ……]
        """
        assert self.online == True
        self.set_order_start(order)
        # driver类中，self.order = order，将该司机的order设置传入的参数order
        self.onservice = True
        self.node.idle_driver_num -= 1

    def set_order_list_start(self, order_list):  # 针对多个订单
        self.order_list = order_list

    def take_order_list(self, order_list):  # 针对多个订单
        assert self.online == True
        self.take_order_list(order_list)
        self.onservice = True
        self.node.idle_driver_num -= 1


    def status_control_eachtime(self, city): # 这里需要改
        # driver-->order--->order_end_time(这个司机served的订单结束时间)

        assert self.city_time == city.city_time # 判断司机的city_time是否等于城市的city_time
        if self.onservice is True:
            assert self.online is True
            order_end_time = self.order.get_assigned_time() + self.order.get_duration()
            # get_assigned_time 是order的分派的时间，属于order类
            # get_duration 是order的持续时间，属于order类
            if self.city_time == order_end_time:  # 服务的订单结束时间等于当前env时间，即司机的一次服务结束
                self.set_position(self.order.get_end_position())
                # driver司机的结束位置等于订单的目的地
                self.set_order_finish()
                # 司机driver onservice设置为false
                self.node.add_driver(self._driver_id, self)
                # 这里的node属于driver类，表示当前driver在这个node内
                city.n_drivers += 1
                # 空闲司机数加一
            elif self.city_time < order_end_time:  # 订单还在持续中，没有完成
                pass
            else:
                raise ValueError('Driver: status_control_eachtime(): order end time less than city time')