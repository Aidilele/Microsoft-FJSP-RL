# from function import cmax
import numpy as np


class StartNode:
    def __init__(self):
        self.sub_node_list = []
        self.disjunction_sub_node_list = []
        self.NodeType = "Start"
        self.p_t = 0
        self.s_w = 0
        self.t_w = 1
        self.apt = 0
        self.p_o = 0
        self.j_o = 0
        self.assign_flag = True

    def add_job(self, operation):
        self.sub_node_list.append(operation)

    def reset(self):
        self.disjunction_sub_node_list = []

    def get_fea(self, info=None):  # 返回状态值，-1返回static，1返回dynamic，0返回all of feature
        critical = 1
        fea = np.array([
            self.apt,
            self.j_o,
            self.p_t,
            self.s_w,
            self.t_w,
            self.p_o,
            critical,
        ])
        # fea = np.array([
        #     self.p_t / self.apt,
        #     self.p_t / info['cmax'],
        #     self.s_w / info['cmax'],
        #     (info['cmax'] - self.p_t - self.s_w) / info['cmax'],
        # ])
        return fea


class EndNode:
    def __init__(self):
        self.pre_node_list = []
        self.disjunction_pre_node_list = []
        self.NodeType = 'End'
        self.t_w = 0
        self.p_t = 0
        self.s_w = 1
        self.apt = 0
        self.p_o = 1
        self.j_o = 1
        self.assign_flag = True

    def add_job(self, operation):
        self.pre_node_list.append(operation)

    def reset(self):
        self.disjunction_pre_node_list = []

    def get_fea(self, info=None):  # 返回状态值，-1返回static，1返回dynamic，0返回all of feature
        critical = 1
        fea = np.array([
            self.apt,
            self.j_o,
            self.p_t,
            self.s_w,
            self.t_w,
            self.p_o,
            critical,
        ])
        # fea = np.array([
        #     self.p_t / self.apt,
        #     self.p_t / info['cmax'],
        #     self.s_w / info['cmax'],
        #     (info['cmax'] - self.p_t - self.s_w) / info['cmax'],
        # ])
        return fea


class MachineNode:
    def __init__(self, machine_index):
        self.location = machine_index  # 该机器的唯一标识符
        self.ope_process_queue = []
        self.operation_list = []
        self.NodeType = 'Machine'
        self.operation_index_list = []
        '''
        number of process operation
        number of processing operation
        idle rate
        '''

    def reset(self):
        self.ope_process_queue = []

    def get_feature(self, flag=0):
        static_feature = np.array([len(self.operation_list)])

        process_time_assume = 0
        for operation in self.ope_process_queue:
            process_time_assume += operation.process_time
        if len(self.ope_process_queue) == 0:
            idle_rate = 0
        else:
            idle_rate = process_time_assume / (self.ope_process_queue[-1].process_end_time - self.ope_process_queue[
                0].process_start_time)
        dynamic_feature = np.array(
            [
                len(self.ope_process_queue),
                idle_rate,
            ]
        )
        if flag == 0:
            return static_feature, dynamic_feature
        elif flag == -1:
            return static_feature
        elif flag == 1:
            return dynamic_feature


class OperationNode:
    def __init__(self, job_index, operation_index, adj_index, machine_index_list):
        self.location = (job_index, operation_index)  # 该工序的唯一标识符
        self.adj_index = adj_index  # 该工序在adj中的索引值
        self.process_machine = None
        self.m_loca = None  # None表示未指派任何机器加工该节点
        self.p_o = None
        self.p_sum = None
        self.p_t = None
        self.s_w = None
        self.t_w = None
        self.dis_pre = None
        self.dis_sub = None
        self.busy_machine_flag = 0
        self.assign_flag = False
        self.machine_index_list = machine_index_list
        self.pre = None
        self.sub = None
        self.j_o = None
        self.j_sum = None
        self.apt = None
        self.NodeType = 'Operation'

        '''
        number of process machine
        number of pre
        number of sub
        '''
        self.static_feature = np.zeros(1)
        '''
        process time
        s_w
        s_w
        '''
        self.dynamic_feature = np.zeros(3)

    def reset(self):
        self.process_machine = None
        self.m_loca = None  # 表示未指派任何机器加工该节点
        self.p_o = None
        self.p_sum = None
        self.p_t = 0
        self.s_w = None
        self.t_w = None
        self.dis_pre = None
        self.dis_sub = None
        self.busy_machine_flag = 0
        self.assign_flag = False

    def get_fea(self, info=None):
        critical = 0
        if self.s_w + self.t_w + self.p_t == info['cmax']:
            critical = 1
        fea = np.array([
            self.apt/info['cmax'],
            (self.j_o+1) / (self.j_sum+1),
            self.p_t / info['cmax'],
            self.s_w / info['cmax'],
            self.t_w / info['cmax'],
            (self.p_o+1) / (self.p_sum+1),
            critical,
        ])
        # fea = np.array([
        #     self.p_t / self.apt,
        #     self.p_t / info['cmax'],
        #     self.s_w / info['cmax'],
        #     (info['cmax'] - self.p_t - self.s_w) / info['cmax'],
        # ])
        return fea
