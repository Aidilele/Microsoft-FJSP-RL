import numpy as np

import RawDataProcess
from Node import *
import random
from copy import deepcopy
import numpy
from UsefulFunction import check_circle
from RawDataProcess import single_process


class FJSPGraph:

    def __init__(self, instance_path):
        instance_data = single_process(instance_path, dict_mode=False)
        self.struct_data = instance_data[0]
        self.machine_num = instance_data[1]
        self.job_num = instance_data[2]
        self.ope_num = instance_data[3]
        self.ope_fea_dim = 7
        self.ope_attr_dim = 5
        # -2--->StartNode -1--->EndNode
        self.adj2location = {}
        self.location2adj = {}
        adj_count = 0
        self.instance = []
        for job_index in range(len(self.struct_data)):
            self.instance.append([])
            for ope_index in range(len(self.struct_data[job_index])):
                self.instance[job_index].append((job_index, ope_index))
                self.adj2location[adj_count] = (job_index, ope_index)
                self.location2adj[(job_index, ope_index)] = adj_count
                adj_count += 1
        self.adj2location[adj_count] = -2
        self.adj2location[adj_count + 1] = -1

        self.ope_ope_adj = np.zeros((self.ope_num + 2, self.ope_num + 2), dtype=np.int32)  # 存储ope之间的job约束
        self.job_sub_pre = np.zeros((self.ope_num, 2), dtype=np.int32)  # 存储每个job待处理的第一个节点
        self.ope_mach_process_time = np.zeros((self.ope_num + 2, self.machine_num))  # 存储ope与对应mach的加工时间
        self.ope_mach_adj = np.zeros((self.ope_num, self.machine_num), dtype=np.int32)  # 存储ope与对应mach的约束关系
        self.ave_process_time = np.zeros(self.ope_num + 2)  # 存储ope的平均处理时间
        self.job_order = np.zeros(self.ope_num + 2)
        self.job_order[-1] = 1.0

        self.current_job_ope = np.zeros((self.job_num), dtype=np.int32)
        self.job_dynamic_adj = np.zeros((self.ope_num + 2, self.ope_num + 2), dtype=np.int32)  # 存储已经分配的ope前后驱
        self.ope_ope_dynamic_adj = np.zeros((self.ope_num + 2, self.ope_num + 2), dtype=np.int32)  # 存储ope之间的machine约束
        self.ope_mach_dynamic_adj = np.zeros(
            (self.ope_num, self.machine_num))  # 存储ope与mach之间的从属关系，转置后可以得到某台mach加工了哪些ope
        self.ope_raw_fea = np.zeros((self.ope_num + 2, self.ope_fea_dim))  # 存储ope节点的原始特征
        self.mach_sub_pre = np.zeros((self.ope_num, 2), dtype=np.int32)  # 存储ope在mach上的前后驱
        self.ope_assign = np.zeros(self.ope_num, dtype=np.int32)  # 存储ope的分配信息
        self.ope_process_mach = np.zeros(self.ope_num + 2, dtype=np.int32)  # 存储ope在哪一台mach上加工

        adj_count = 0
        for job_index in range(len(self.struct_data)):
            for ope_index in range(len(self.struct_data[job_index])):
                self.ope_mach_process_time[adj_count] = np.array(
                    self.struct_data[job_index][ope_index])
                self.ave_process_time[adj_count] = self.ope_mach_process_time[adj_count][
                    np.where(self.ope_mach_process_time[adj_count] > 0)].mean()
                self.ope_mach_adj[adj_count][np.where(self.ope_mach_process_time[adj_count] > 0)] = 1
                self.job_order[adj_count] = (ope_index + 1) / (len(self.struct_data[job_index]) + 1)

                adj_count += 1

        for job_index in range(len(self.struct_data)):
            self.ope_ope_adj[self.location2adj[(job_index, 0)]][-2] = -1
            for ope_index in range(1, len(self.struct_data[job_index])):
                self.ope_ope_adj[self.location2adj[(job_index, ope_index)]][
                    self.location2adj[(job_index, ope_index)] - 1] = -1
            self.ope_ope_adj[-1][self.location2adj[(job_index, ope_index)]] = -1

        for job_index in range(len(self.struct_data)):
            self.ope_ope_adj[-2][self.location2adj[(job_index, 0)]] = 1
            for ope_index in range(0, len(self.struct_data[job_index]) - 1):
                self.ope_ope_adj[self.location2adj[(job_index, ope_index)]][
                    self.location2adj[(job_index, ope_index)] + 1] = 1
            self.ope_ope_adj[self.location2adj[(job_index, ope_index)]][-1] = 1

        self.job_sub_pre[:, 1] = np.argmax(self.ope_ope_adj, 1)[:-2]
        self.job_sub_pre[:, 0] = np.argmin(self.ope_ope_adj, 1)[:-2]

    def opelist2adj(self, opelist):

        adj = np.zeros((self.ope_num + 2, self.ope_num + 2), dtype=np.int32)
        for mach in opelist:
            pre_ope = -2
            for ope in mach:
                current_ope = self.location2adj[ope]
                adj[current_ope][pre_ope] = -1
                adj[pre_ope][current_ope] = 1
                pre_ope = current_ope
            current_ope = -1
            adj[current_ope][pre_ope] = -1
            adj[pre_ope][current_ope] = 1

        return adj

    def get_fea(self, solution=0):
        if solution == 0:
            solution = []
            for i in range(self.machine_num):
                solution.append([])
        self.solution = solution
        ope_fea = np.zeros((self.ope_num + 2, self.ope_fea_dim))
        '''
            0:assigned
            1:start time 
            2:process time
            3:finsh time
            4:job order
            5:mach order
            6:avenger process time
        '''
        ope_fea[:, 6] = self.ave_process_time  # avenger process time
        ope_fea[:, 4] = self.job_order  # job order
        ope_fea[:-2, 5] = 1.0
        ope_fea[-1, 5] = 1.0
        ope_fea[(-1, -2), 0] = 1
        mach_dynamic_adj = self.opelist2adj(solution)
        machine_index = 0
        for mach in solution:
            mach_order = 1
            for ope in mach:
                ope_adj = self.location2adj[ope]
                ope_fea[ope_adj, 0] = 1  # assigned
                ope_fea[ope_adj, 2] = self.ope_mach_process_time[ope_adj, machine_index]  # process time
                ope_fea[ope_adj, 5] = mach_order / (len(ope) + 1)  # mach order
            machine_index += 1

        # instance = []
        # for job in self.instance:
        #     job_ope = []
        #     for ope in job:
        #         if ope_fea[self.location2adj[ope], 0] == 1:
        #             job_ope.append(ope)
        #         else:
        #             break
        #     instance.append(job_ope)
        # job_dynamic_adj = self.opelist2adj(instance)
        mach_sub_pre = np.zeros([self.ope_num, 2], dtype=np.int32)
        mach_sub_pre[:, 1] = np.argmax(mach_dynamic_adj, 1)[:-2]
        mach_sub_pre[:, 0] = np.argmin(mach_dynamic_adj, 1)[:-2]
        # job_sub_pre = np.zeros([self.ope_num + 2, 2], dtype=np.int32)
        # job_sub_pre[:, 1] = np.argmax(job_dynamic_adj, 1)[:-2]
        # job_sub_pre[:, 0] = np.argmin(job_dynamic_adj, 1)[:-2]

        # start time
        ope_fea[:, 1] = -1
        ope_fea[-2, 1] = 0
        finsh_ope = 0
        job_finsh_index = [0] * self.job_num
        while finsh_ope < self.ope_num:
            job_index = 0
            for job in self.instance:
                for ope in job[job_finsh_index[job_index]:]:
                    ope_adj = self.location2adj[ope]
                    ope_job_pre = self.job_sub_pre[ope_adj, 0]
                    ope_job_pre_sw = ope_fea[ope_job_pre, 1]
                    ope_job_pre_pt = ope_fea[ope_job_pre, 2]
                    if ope_fea[ope_adj, 0] == 0:
                        ope_mach_pre_sw = 0
                        ope_mach_pre_pt = -1
                        ope_fea[ope_adj, 2] = ope_fea[ope_adj, 6]
                    else:
                        ope_mach_pre = mach_sub_pre[ope_adj, 0]
                        ope_mach_pre_sw = ope_fea[ope_mach_pre, 1]
                        ope_mach_pre_pt = ope_fea[ope_mach_pre, 2]
                    if ope_job_pre_sw >= 0 and ope_mach_pre_sw >= 0:
                        ope_fea[ope_adj, 1] = max(ope_job_pre_pt + ope_job_pre_sw,
                                                  ope_mach_pre_pt + ope_mach_pre_sw)
                        finsh_ope += 1
                        job_finsh_index[job_index] += 1
                job_index += 1
        ope_fea[:, 3] = ope_fea[:, 1] + ope_fea[:, 2]  # finsh time
        true_cmax = np.max(ope_fea[np.where(ope_fea[:, 0] > 0), 3])
        # cmax=true_cmax/ope_fea[:,0].sum()*ope_fea.shape[0]
        # cmax = true_cmax + ope_fea[np.where(ope_fea[:, 0] == 0), 6].sum()

        unassigned_ope = np.where(ope_fea[:-2, 0] == 0)
        assigned_ope = np.where(ope_fea[:-2, 0] == 1)
        cmax_array = np.zeros((self.ope_num,self.machine_num))
        cmax_array[:] = self.ope_mach_process_time[:-2]
        cmax_array[unassigned_ope] /= self.ope_mach_adj[unassigned_ope].sum(1).reshape(-1, 1)
        cmax_array[assigned_ope] = 0
        cmax = np.max(cmax_array.sum(0))
        cmax += true_cmax

        ope_fea[-1, (1, 3)] = np.max(ope_fea[:, 3])
        pre_cmax = ope_fea[-1, 3]
        ope_fea[:, [1, 2, 3]] /= pre_cmax

        ope_select = -np.ones(self.job_num, dtype=np.int32)
        action_mask = np.zeros((self.job_num, self.machine_num))
        job_select = [-1] * self.job_num
        job_index = 0
        for job in self.instance:
            for ope in job:
                if ope_fea[self.location2adj[ope], 0] == 0:
                    job_select[job_index] = ope
                    ope_select[job_index] = self.location2adj[ope]
                    break
            job_index += 1
        self.job_select = job_select

        for job_index in range(self.job_num):
            ope = job_select[job_index]
            if job_select[job_index] == -1:
                action_mask[job_index] = -np.Inf
            else:
                ope_adj = self.location2adj[ope]
                action_mask[job_index, np.where(self.ope_mach_adj[ope_adj] == 0)] = -np.Inf

        action_mask = action_mask.reshape(-1)

        job_adj = np.zeros((self.ope_num + 2, self.job_num), dtype=np.int32)
        mach_adj = np.zeros((self.ope_num + 2, self.machine_num), dtype=np.int32)
        job_index = 0
        for job in self.instance:
            for ope in job:
                job_adj[self.location2adj[ope], job_index] = 1
            job_index += 1
        job_adj[(-1, -2), :] = 1

        mach_index = 0
        for mach in solution:
            for ope in mach:
                mach_adj[self.location2adj[ope], mach_index] = 1
            mach_index += 1
        mach_adj[(-1, -2), :] = 1
        ope_adj = np.abs(self.ope_ope_adj) | np.abs(mach_dynamic_adj)

        return (np.expand_dims(ope_fea, 0),
                np.expand_dims(ope_adj, 0),
                np.expand_dims(job_adj.T, 0),
                np.expand_dims(mach_adj.T, 0),
                np.expand_dims(action_mask, 0),
                cmax,
                np.expand_dims(ope_select, 0)
                )

    def excute_action(self, action):
        solution = deepcopy(self.solution)
        target_job = action // self.machine_num
        target_mach = action % self.machine_num
        target_ope = self.job_select[target_job]
        solution[target_mach].append(target_ope)
        return solution

    # def apply_solution(self, solution):
    #
    #     self.solution = solution
    #     mach_index = 0
    #     for mach in solution:
    #         ope_index = 0
    #         for ope in mach:
    #             self.ope_mach_dynamic_adj[self.location2adj[ope]][mach_index] = 1
    #             self.ope_assign[self.location2adj[ope]] = 1
    #             self.ope_process_mach[self.location2adj[ope]] = mach_index
    #             ope_index += 1
    #         mach_index += 1
    #     self.ope_ope_dynamic_adj = self.opelist2adj(solution)
    #     self.mach_sub_pre[:, 1] = np.argmax(self.ope_ope_dynamic_adj, 1)[:-2]
    #     self.mach_sub_pre[:, 0] = np.argmin(self.ope_ope_dynamic_adj, 1)[:-2]
    #
    #     return 0
    #
    # def cal_sw(self):
    #
    #     self.ope_process_time = np.zeros(self.ope_num + 2)
    #     for mach in self.solution:
    #         for ope in mach:
    #             ope_adj = self.location2adj[ope]
    #             self.ope_process_time[ope_adj] = self.ope_mach_process_time[ope_adj][self.ope_process_mach[ope_adj]]
    #
    #     self.ope_sw = -np.ones(self.ope_num + 2)
    #     self.ope_sw[-2] = 0
    #     total_ope = self.ope_assign.sum()
    #     finsh_ope = 0
    #     mach_finsh_index = [0] * self.machine_num
    #     while finsh_ope < total_ope:
    #         mach_index = 0
    #         for mach in self.solution:
    #             for ope in mach[mach_finsh_index[mach_index]:]:
    #                 ope_adj = self.location2adj[ope]
    #                 ope_job_pre = self.job_sub_pre[ope_adj, 0]
    #                 ope_mach_pre = self.mach_sub_pre[ope_adj, 0]
    #                 if self.ope_sw[ope_job_pre] >= 0 and self.ope_sw[ope_mach_pre] >= 0:
    #                     ope_job_pre_pt = self.ope_process_time[ope_job_pre]
    #                     ope_mach_pre_pt = self.ope_process_time[ope_mach_pre]
    #                     self.ope_sw[ope_adj] = max(self.ope_sw[ope_job_pre] + ope_job_pre_pt,
    #                                                self.ope_sw[ope_mach_pre] + ope_mach_pre_pt)
    #                     finsh_ope += 1
    #                     mach_finsh_index[mach_index] += 1
    #             mach_index += 1
    #
    #     self.cmax = 0
    #     for mach in self.solution:
    #         for ope in mach[-1:]:
    #             ope_adj = self.location2adj[ope]
    #             if self.ope_sw[ope_adj] + self.ope_process_time[ope_adj] > self.cmax:
    #                 self.cmax = self.ope_sw[ope_adj] + self.ope_process_time[ope_adj]
    #     return 0


if __name__ == '__main__':
    file_path = './dataset/0503/5j_3m_001.fjs'
    graph = FJSPGraph(file_path)
    solution = [[(4, 0), (2, 0), (1, 1), (4, 2), (3, 0), ],
                [(4, 1), (2, 1), ],
                [(0, 0), (1, 0), (0, 1), (0, 2), (1, 2), ]]
    solution = [[], [], []]
    graph.get_fea(solution)
