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
        struct_data = instance_data[0]
        self.struct_data = instance_data[0]
        self.machine_num = instance_data[1]
        self.job_num = instance_data[2]
        self.ope_num = instance_data[3]

        self.start_node = StartNode()
        self.end_node = EndNode()

        self.machine_list = []  # machine节点从1开始编号
        self.ope_list = []  # operation和job从0开始编号
        self.adj2location = {}
        self.location2adj = {}

        for machine in range(self.machine_num):
            machine_node = MachineNode(machine)
            self.machine_list.append(machine_node)

        adj_index = 0

        for job_index in range(len(struct_data)):
            job_list = []
            for ope_index in range(len(struct_data[job_index])):
                machine_list = []
                for index in range(self.machine_num):
                    if struct_data[job_index][ope_index][index] != -1:
                        machine_list.append(index)
                        self.machine_list[index].operation_index_list.append((job_index, ope_index))
                ope_node = OperationNode(job_index, ope_index, adj_index, machine_list)
                ope_node.j_o = ope_index
                ope_node.j_sum = len(struct_data[job_index])
                ope_node.apt = 0
                for mach_index in ope_node.machine_index_list:
                    ope_node.apt += struct_data[job_index][ope_index][mach_index]
                ope_node.apt = ope_node.apt / len(ope_node.machine_index_list)
                self.adj2location[adj_index] = (job_index, ope_index)
                self.location2adj[(job_index, ope_index)] = adj_index
                adj_index += 1
                job_list.append(ope_node)
            self.ope_list.append(job_list)

        self.ope_mach_adj = np.zeros((self.ope_num, self.machine_num), dtype=np.int64)
        self.ope_ope_adj = np.zeros((self.ope_num, self.ope_num), dtype=np.int64)
        self.ope_ope_dynamic_adj = np.zeros((self.ope_num, self.ope_num), dtype=np.int64)
        self.ope_mach_dynamic_adj = np.zeros((self.ope_num, self.machine_num), dtype=np.int64)

        adj_index = 0
        for job_index in range(len(struct_data)):
            for ope_index in range(len(struct_data[job_index])):
                self.ope_mach_adj[adj_index] = np.array(struct_data[job_index][ope_index])
                adj_index += 1

        adj_index = 0
        for job_index in range(len(struct_data)):
            for ope_index in range(len(struct_data[job_index])):
                if ope_index < (len(struct_data[job_index]) - 1):
                    self.ope_ope_adj[adj_index][adj_index + 1] = 1
                adj_index += 1

        adj_index = 0
        for job_index in range(len(struct_data)):
            for ope_index in range(len(struct_data[job_index])):
                if ope_index > 0:
                    self.ope_ope_adj[adj_index][adj_index - 1] = -1
                adj_index += 1

        sub_arg = np.argmax(self.ope_ope_adj, 1)
        pre_arg = np.argmin(self.ope_ope_adj, 1)
        for job in self.ope_list:
            for ope in job:
                ope.sub = self.adj2location[sub_arg[ope.adj_index]]
                ope.pre = self.adj2location[pre_arg[ope.adj_index]]
            job[0].pre = 'Start'
            job[-1].sub = 'End'

        self.ope_ope_adj_se = np.pad(self.ope_ope_adj, ((1, 1), (1, 1)))
        top = np.zeros(self.ope_num + 2, dtype=np.int64)
        bottom = np.zeros(self.ope_num + 2, dtype=np.int64)
        for job in self.ope_list:
            top[job[0].adj_index + 1] = 1
            bottom[job[-1].adj_index + 1] = -1
        left = -top
        right = -bottom
        self.ope_ope_adj_se[0, :] += top
        self.ope_ope_adj_se[-1, :] += bottom
        self.ope_ope_adj_se[:, 0] += left
        self.ope_ope_adj_se[:, -1] += right

    def check_process_time(self, operation, machine):  # 仅读取数据，不写入数据
        if type(operation) == tuple:
            return self.ope_mach_adj[self.location2adj[operation]][machine]
        else:
            return self.ope_mach_adj[operation][machine]

    def trans2object(self, index):  # 仅读取数据，不写入数据
        if type(index) == tuple:
            return self.ope_list[index[0]][index[1]]
        if type(index) == int:
            return self.machine_list[index]
        if index == 'Start':
            return self.start_node
        if index == 'End':
            return self.end_node
        return index

    def findpre(self, node, graph):
        if type(node) == tuple:
            node = self.location2adj[node]
        pre = np.argmin(graph[node])
        if graph[node][pre] != 0:
            return self.adj2location[pre]
        return 'Start'

    def findsub(self, node, graph):
        if type(node) == tuple:
            node = self.location2adj[node]
        sub = np.argmax(graph[node])
        if graph[node][sub] != 0:
            return self.adj2location[sub]
        return 'End'

    def reset(self):
        '''
        重置实例的solution，只保留各个node间的static关系
        '''
        self.start_node.reset()
        self.end_node.reset()
        self.ope_ope_dynamic_adj = np.zeros((self.ope_num, self.ope_num), dtype=np.int64)
        self.ope_mach_dynamic_adj = np.zeros((self.ope_num, self.machine_num), dtype=np.int64)
        self.solution=[]
        for machine in self.machine_list:
            machine.reset()
            self.solution.append([])
        for job in self.ope_list:
            for ope in job:
                ope.reset()
        return 0

    def apply_solution(self, solution):
        self.reset()

        for m_i in range(self.machine_num):
            self.trans2object(m_i).ope_process_queue = solution[m_i]
            for ope_i in range(len(solution[m_i])):
                ope = self.trans2object(solution[m_i][ope_i])
                self.ope_mach_dynamic_adj[self.location2adj[solution[m_i][ope_i]]][m_i] = 1
                ope.process_machine = m_i
                ope.p_t = self.check_process_time(solution[m_i][ope_i], m_i)
                if ope.p_t == -1:
                    print(0)
                ope.p_o = ope_i
                ope.p_sum = len(solution[m_i])
                ope.m_loca = (m_i, ope_i)
                ope.assign_flag = True
                if ope_i < len(solution[m_i]) - 1:
                    self.ope_ope_dynamic_adj[
                        self.location2adj[solution[m_i][ope_i]]][self.location2adj[solution[m_i][ope_i + 1]]] = 1
        for m_i in range(self.machine_num):
            for ope_i in range(len(solution[m_i])):
                if ope_i > 0:
                    self.ope_ope_dynamic_adj[
                        self.location2adj[solution[m_i][ope_i]]][self.location2adj[solution[m_i][ope_i - 1]]] = -1
        self.ope_ope_dynamic_adj_se = np.pad(self.ope_ope_dynamic_adj, ((1, 1), (1, 1)))
        top = np.zeros(self.ope_num + 2, dtype=np.int64)
        bottom = np.zeros(self.ope_num + 2, dtype=np.int64)
        for mach in solution:
            if len(mach) > 0:
                top[self.trans2object(mach[0]).adj_index + 1] = 1
                bottom[self.trans2object(mach[-1]).adj_index + 1] = -1
            else:
                top[-1] = 1
                bottom[0] = -1
        left = -top
        right = -bottom
        self.ope_ope_dynamic_adj_se[0, :] |= top
        self.ope_ope_dynamic_adj_se[-1, :] |= bottom
        self.ope_ope_dynamic_adj_se[:, 0] |= left
        self.ope_ope_dynamic_adj_se[:, -1] |= right

        check_s_graph = np.zeros((self.ope_num, self.ope_num), dtype=np.int8)
        check_d_graph = np.zeros((self.ope_num, self.ope_num), dtype=np.int8)
        for i in range(self.ope_num):
            for j in range(self.ope_num):
                check_d_graph[i][j] = int(max(0, self.ope_ope_dynamic_adj[i][j]))
                check_s_graph[i][j] = int(max(0, self.ope_ope_adj[i][j]))

        check_graph = check_s_graph | check_d_graph
        check_result = check_circle(check_graph)
        if check_result == 0:
            self.solution = solution
            sub_arg = np.argmax(self.ope_ope_dynamic_adj, 1)
            pre_arg = np.argmin(self.ope_ope_dynamic_adj, 1)
            for machine in solution:
                if len(machine) == 0:
                    continue
                for ope in machine:
                    operation = self.trans2object(ope)
                    operation.dis_sub = self.adj2location[sub_arg[operation.adj_index]]
                    operation.dis_pre = self.adj2location[pre_arg[operation.adj_index]]
                self.trans2object(machine[0]).dis_pre = 'Start'
                self.trans2object(machine[-1]).dis_sub = 'End'
        else:
            self.apply_solution(self.solution)
        return check_result

    def cal_sw_tw(self):
        def cal_sw(node):
            if node.s_w != None:
                return node.s_w
            if node.assign_flag == False:
                node.s_w = 0
                return 0
            else:
                node_pre = self.trans2object(node.pre)
                node_dis_pre = self.trans2object(node.dis_pre)
                if node_pre.s_w != None and node_dis_pre.s_w != None:
                    value = max(node_dis_pre.s_w + node_dis_pre.p_t,
                                node_pre.s_w + node_pre.p_t)

                elif node_pre.s_w == None and node_dis_pre.s_w != None:
                    value = max(node_dis_pre.s_w + node_dis_pre.p_t,
                                cal_sw(node_pre) + node_pre.p_t)

                elif node_pre.s_w != None and node_dis_pre.s_w == None:
                    value = max(cal_sw(node_dis_pre) + node_dis_pre.p_t,
                                node_pre.s_w + node_pre.p_t)

                elif node_pre.s_w == None and node_dis_pre.s_w == None:
                    value = max(cal_sw(node_dis_pre) + node_dis_pre.p_t,
                                cal_sw(node_pre) + node_pre.p_t)
                node.s_w = value
                return value

        def cal_tw(node):
            if node.t_w != None:
                return node.t_w
            if node.assign_flag == False:
                node.t_w = 0
                return 0
            else:
                node_sub = self.trans2object(node.sub)
                node_dis_sub = self.trans2object(node.dis_sub)
                if node_sub.t_w != None and node_dis_sub.t_w != None:
                    value = max(node_dis_sub.t_w + node_dis_sub.p_t,
                                node_sub.t_w + node_sub.p_t)

                elif node_sub.t_w == None and node_dis_sub.t_w != None:
                    value = max(node_dis_sub.t_w + node_dis_sub.p_t,
                                cal_tw(node_sub) + node_sub.p_t)

                elif node_sub.t_w != None and node_dis_sub.t_w == None:
                    value = max(cal_tw(node_dis_sub) + node_dis_sub.p_t,
                                node_sub.t_w + node_sub.p_t)

                elif node_sub.t_w == None and node_dis_sub.t_w == None:
                    value = max(cal_tw(node_dis_sub) + node_dis_sub.p_t,
                                cal_tw(node_sub) + node_sub.p_t)
                node.t_w = value
                return value

        # for job in self.ope_list:
        #     for ope in job:
        #         cal_sw(ope)
        #
        # for job in self.ope_list:
        #     for ope in job:
        #         cal_tw(ope)
        for mach in self.solution:
            for ope in mach:
                cal_sw(self.trans2object(ope))
        for mach in self.solution:
            for ope in mach:
                cal_tw(self.trans2object(ope))

    def makespan(self):
        cmax = 0
        for machine in self.solution:
            if len(machine) > 0:
                ope = self.trans2object(machine[-1])
                ope_max = ope.s_w + ope.p_t + ope.t_w
                cmax = max(cmax, ope_max)
        return cmax

    def get_critical_node(self):  # 仅读取数据，不写入数据
        cmax = self.makespan()
        critical_node = []
        for job in self.ope_list:
            for ope in job:
                if ope.s_w + ope.p_t + ope.t_w == cmax:
                    critical_node.append(ope.location)
        return critical_node

    def get_critical_path(self):
        critical_path = []
        cmax = self.makespan()
        for machine in self.solution:
            if self.trans2object(machine[-1]).s_w + self.trans2object(machine[-1]).p_t + self.trans2object(
                    machine[-1]).t_w == cmax:
                last_node = machine[-1]
                break
        while last_node != 'Start':
            critical_path.insert(0, last_node)
            if self.trans2object(self.trans2object(last_node).pre).s_w + self.trans2object(
                    self.trans2object(last_node).pre).p_t == self.trans2object(last_node).s_w:
                last_node = self.trans2object(last_node).pre
            else:
                last_node = self.trans2object(last_node).dis_pre
        return critical_path

    def get_random_node(self):
        adj_index = random.randint(0, self.ope_num - 1)
        return [self.adj2location[adj_index]]

    def generate_bottom_solution(self):

        def greedy_pick(ope):
            target_machine = ope.machine_index_list[0]
            for machine in ope.machine_index_list:
                if self.check_process_time(ope.adj_index, machine) < self.check_process_time(ope.adj_index,
                                                                                             target_machine):
                    target_machine = machine
            return target_machine

        solution = []
        for i in range(self.machine_num):
            solution.append([])

        for job in self.ope_list:
            node = job[-1]
            node.bottom_level = node.apt
            node = self.trans2object(node.pre)
            while node.NodeType != 'Start':
                node.bottom_level = self.trans2object(node.sub).bottom_level + node.apt
                node = self.trans2object(node.pre)
        job_num = self.job_num
        machine_num = self.machine_num
        operation_num = self.ope_num
        finsh_num = 0
        job_finsh = [0] * job_num
        while finsh_num < operation_num:
            opt_job = 0
            opt_bottom_level = 0
            for current_job in range(job_num):
                if job_finsh[current_job] >= len(self.ope_list[current_job]):
                    continue
                if self.ope_list[current_job][job_finsh[current_job]].bottom_level > opt_bottom_level:
                    opt_bottom_level = self.ope_list[current_job][job_finsh[current_job]].bottom_level
                    opt_job = current_job
            machine_index = greedy_pick(self.ope_list[opt_job][job_finsh[opt_job]])
            solution[machine_index].append((opt_job, job_finsh[opt_job]))
            job_finsh[opt_job] = job_finsh[opt_job] + 1
            finsh_num += 1
        return solution

    def generate_greedy_solution(self):
        solution = []

        def choose_mach(mach_list, solution):
            target_machine = mach_list[0]
            for mach in mach_list:
                if len(solution[target_machine]) > len(solution[mach]):
                    target_machine = mach
            return target_machine

        for i in range(self.machine_num):
            solution.append([])

        for job in self.ope_list:
            for ope in job:
                target_mach = choose_mach(ope.machine_index_list, solution)
                solution[target_mach].append(ope.location)
        return solution

    def generate_random_solution(self):
        solution = []
        for i in range(self.machine_num):
            solution.append([])
        job_current_ope = [0] * self.job_num
        enble_job = [x for x in range(self.job_num)]

        for ope_index in range(self.ope_num):
            job_index = random.sample(enble_job, 1)[0]
            enble_mach = self.ope_list[job_index][job_current_ope[job_index]].machine_index_list
            mach_index = random.sample(enble_mach, 1)[0]
            solution[mach_index].append(self.ope_list[job_index][job_current_ope[job_index]].location)
            job_current_ope[job_index] = job_current_ope[job_index] + 1
            if job_current_ope[job_index] == len(self.ope_list[job_index]):
                enble_job.remove(job_index)
        return solution

    def get_fea(self, ope_fea_dim=6, insert_fea_dim=4):
        info = {}
        info['cmax'] = self.makespan()
        ope_fea = np.zeros([self.ope_num + 2, ope_fea_dim])
        self.action_dict = {}
        action_index = 0
        assign_mask = np.ones([self.ope_num + 2, self.ope_num + 2], dtype=np.int64)

        for job in self.ope_list:
            for ope in job:
                if ope.assign_flag == False:
                    assign_mask[ope.adj_index + 1, :] = 0
                    assign_mask[:, ope.adj_index + 1] = 0
        for mach in self.solution:
            for ope_index in mach:
                ope = self.trans2object(ope_index)
                ope_fea[ope.adj_index + 1] = ope.get_fea(info)
                self.action_dict[action_index] = ope_index
                action_index += 1

        # ope_fea[:, 0] = ope_fea[:, 0] / np.max(ope_fea[:, 0])  #是否对feature的每一维归一化
        ope_adj = np.abs(self.ope_ope_adj_se).astype(np.int8) | np.abs(self.ope_ope_dynamic_adj_se).astype(np.int8)
        ope_adj = ope_adj + np.eye(ope_fea.shape[0], dtype=np.int64)
        ope_adj *= assign_mask

        insert_adj = np.zeros([self.ope_num + self.machine_num, self.ope_num + 2])
        for machine in self.solution:
            before = 0
            for ope in machine:
                after = self.location2adj[ope] + 1
                insert_adj[self.location2adj[ope]][before] = 1
                insert_adj[self.location2adj[ope]][after] = 1
                before = after

        insert_index = self.ope_num
        for machine in self.solution:
            if len(machine) > 0:
                before = self.location2adj[machine[-1]] + 1
            else:
                before = 0
            after = -1
            insert_adj[insert_index][before] = 1
            insert_adj[insert_index][after] = 1
            insert_index += 1
        # insert_fea=np.zeros([self.ope_num + self.machine_num, insert_fea_dim])
        # for machine in self.solution:
        #     if len(machine) > 0:
        #         fea=np.array([
        #             self.trans2object(machine[0]),#pre_node_sw+pt
        #             0,#sub_node_sw
        #         ])
        return np.expand_dims(ope_fea, 0), np.expand_dims(ope_adj, 0), np.expand_dims(insert_adj, 0)

    def get_insert_mask(self):
        mask = []
        cmax = self.makespan()

        def find_RL(s_, t_, machine):
            R = 0
            for ope in machine:
                if self.trans2object(ope).s_w + self.trans2object(ope).p_t <= s_:
                    R += 1
                else:
                    break
            L = 0
            for ope in machine:
                if self.trans2object(ope).t_w + self.trans2object(ope).p_t > t_:
                    L += 1
                else:
                    break
            return min(R, L), max(R, L)

        for job in self.ope_list:
            for ope in job:
                ope_mask = np.ones(self.ope_num + self.machine_num) * (-np.inf)

                if ope.assign_flag and ope.s_w + ope.t_w + ope.p_t == cmax:
                    s_ = self.trans2object(ope.pre).s_w + self.trans2object(ope.pre).p_t
                    t_ = self.trans2object(ope.sub).t_w + self.trans2object(ope.sub).p_t
                    machine_index = 0
                    for machine in self.solution:
                        if machine_index in ope.machine_index_list:
                            left, right = find_RL(s_, t_, machine)
                            if right == len(machine):
                                ope_mask[self.ope_num + machine_index] = 0
                            for insert in range(left, right):
                                ope_mask[self.location2adj[machine[insert]]] = 0
                        machine_index += 1
                # ope_mask[ope.adj_index] = -np.inf  # 屏蔽当前位置防止原地不动
                mask.append(ope_mask)
        mask = np.concatenate(mask)

        # 屏蔽当前位置防止原地不动
        mach_index = 0
        for mach in self.solution:
            if len(mach)>0:
                mask_index = self.location2adj[mach[-1]] * (self.ope_num + self.machine_num) + self.ope_num + mach_index
                mask[mask_index] = -np.inf

        return np.expand_dims(mask, 0)

    def excute_action(self, action):

        res_solution = deepcopy(self.solution)
        ope_adj = action // (self.ope_num + self.machine_num)
        postion = action % (self.ope_num + self.machine_num)
        ope = self.adj2location[ope_adj]
        res_solution[self.trans2object(ope).process_machine].__delitem__(self.trans2object(ope).p_o)
        if postion >= self.ope_num:
            res_solution[postion - self.ope_num].insert(len(res_solution[postion - self.ope_num]), ope)
        else:
            insert_machine = self.trans2object(self.adj2location[postion]).process_machine
            insert_position = self.trans2object(self.adj2location[postion]).p_o
            res_solution[insert_machine].insert(insert_position, ope)
        self.apply_solution(res_solution)
        self.cal_sw_tw()
        return self.makespan()

    def trans2schedule(self):

        s_w_list = [np.inf]*self.ope_num
        assgin_count=0
        for job in self.ope_list:
            for ope in job:
                if ope.assign_flag==True:
                    assgin_count+=1
                    s_w_list[ope.adj_index]=ope.s_w
        sort_index = sorted(range(len(s_w_list)), key=lambda x: s_w_list[x], reverse=False)
        schedule = np.zeros((assgin_count, 3), dtype=np.int64)
        for i in range(assgin_count):
            current_ope = self.trans2object(self.adj2location[sort_index[i]])
            schedule[i][0] = current_ope.adj_index
            schedule[i][1] = current_ope.process_machine
            schedule[i][2] = current_ope.location[0]
        return schedule


if __name__ == '__main__':
    file_path = './dataset/0503/5j_3m_001.fjs'
    for i in range(1):
        graph = FJSPGraph(file_path)
        greedy_solution = [[],
                           [(2, 0)],
                           [(4, 0)]]
        greedy_solution = graph.generate_bottom_solution()
        graph.apply_solution(greedy_solution)
        graph.cal_sw_tw()
        for job in graph.ope_list:
            for ope in job:
                print(ope.s_w)
        for j in range(20):
            fea = graph.get_fea()
            mask = graph.get_insert_mask()
            cmax = graph.makespan()
            actions=[]
            for k in range(mask[0].shape[0]):
                if mask[0][k]==0:
                    actions.append(k)
            action=random.sample(actions,1)[0]
            graph.excute_action(action)
