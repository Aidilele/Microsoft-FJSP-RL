from Graph import *
from copy import deepcopy
from GanttChart import GanttChart
import random
import os
import matplotlib.pyplot as plt
import numpy as np


class CrossSwap:

    def __init__(self, type="Critical"):
        self.type = type

    def apply(self, graph):
        all_solution = []
        graph.cal_sw_tw()
        cmax = graph.makespan()
        current_info = {
            'cmax': cmax,
            'solution': graph.solution,
            'source_node': None,
            'target_node': None,
        } ,
        node_pro_list = graph.get_critical_node()
        current_solution = graph.solution
        for node in node_pro_list:
            node_obj = graph.trans2object(node)
            for machine in node_obj.machine_index_list:
                machine_obj = graph.trans2object(machine)
                for m_node in machine_obj.ope_process_queue:
                    m_node_obj = graph.trans2object(m_node)
                    if node_obj.process_machine in m_node_obj.machine_index_list:
                        new_solution = deepcopy(current_solution)
                        new_solution[node_obj.m_loca[0]][node_obj.m_loca[1]] = m_node
                        new_solution[m_node_obj.m_loca[0]][m_node_obj.m_loca[1]] = node
                        test_graph = deepcopy(graph)
                        result = test_graph.apply_solution(new_solution)
                        if result == 0:
                            test_graph.cal_sw_tw()
                            cmax = test_graph.makespan()
                            if cmax < current_info['cmax']:
                                current_info['cmax'] = cmax
                                current_info['solution'] = new_solution
                                current_info['source_node'] = node
                                current_info['target_node'] = m_node
        current_info['log'] = 'Swap:' + str(current_info['source_node']) + '-->' + str(
            current_info['target_node']) + '||cmax:' + str(current_info['cmax'])
        return [current_info['solution']], 'CrossSwap'

    def __call__(self, *args, **kwargs):
        return self.apply(args[0])


class GreedyCrossSwap:

    def __init__(self, type="Critical"):
        self.type = type

    def apply(self, graph):
        def find_RL(s_, t_, machine):
            R = 0
            for ope in machine:
                if graph.trans2object(ope).s_w + graph.trans2object(ope).p_t <= s_:
                    R += 1
                else:
                    break
            L = 0
            for ope in machine:
                if graph.trans2object(ope).t_w + graph.trans2object(ope).p_t > t_:
                    L += 1
                else:
                    break
            return L, R

        all_solution = []
        graph.cal_sw_tw()
        cmax = graph.makespan()
        current_info = {
            'cmax': cmax,
            'solution': graph.solution,
            'source_node': None,
            'target_node': None,
        }
        node_pro_list = graph.get_critical_node()
        for current_node in node_pro_list:
            # res_solution = deepcopy(graph.solution)
            # res_solution[graph.trans2object(current_node).process_machine].__delitem__(
            #     graph.trans2object(current_node).p_o)
            current_machine = graph.trans2object(current_node).process_machine
            s_ = graph.trans2object(graph.trans2object(current_node).pre).s_w + graph.trans2object(
                graph.trans2object(current_node).pre).p_t
            t_ = graph.trans2object(graph.trans2object(current_node).sub).t_w + graph.trans2object(
                graph.trans2object(current_node).sub).p_t
            for mach_index in range(graph.machine_num):
                if mach_index == current_machine:
                    continue
                pvk = graph.check_process_time(current_node, mach_index)
                if pvk <= 0:
                    continue
                else:

                    L, R = find_RL(s_, t_, graph.solution[mach_index])
                    best_node = 0
                    if L < R:
                        min_tpt = float('inf')
                        for target_node in graph.solution[mach_index][L:R]:
                            tpt = graph.check_process_time(target_node, current_machine)
                            if tpt > 0 and tpt < min_tpt:
                                min_tpt = tpt
                                best_node = target_node


                    elif L > R:
                        for target_node in graph.solution[mach_index][R:L]:
                            if graph.check_process_time(target_node, current_machine) > 0:
                                sj_ = graph.trans2object(graph.trans2object(target_node).pre).s_w + graph.trans2object(
                                    graph.trans2object(target_node).pre).p_t
                                tj_ = graph.trans2object(graph.trans2object(target_node).sub).t_w + graph.trans2object(
                                    graph.trans2object(target_node).sub).p_t
                                Lj, Rj = find_RL(sj_, tj_, graph.solution[current_machine])
                                J = graph.trans2object(target_node).p_o - 1
                                if J < max(Lj, Rj):
                                    best_node = target_node
                                    break
                    if best_node != 0:
                        new_solution = deepcopy(graph.solution)
                        best_mach = graph.trans2object(best_node).process_machine
                        best_pos = graph.trans2object(best_node).p_o
                        source_mach = current_machine
                        source_pos = graph.trans2object(current_node).p_o
                        new_solution[best_mach][best_pos] = current_node
                        new_solution[source_mach][source_pos] = best_node
                        all_solution.append(new_solution)
        return all_solution, 'CriticalCrossSwap'

    def __call__(self, *args, **kwargs):
        return self.apply(args[0])


class GreedyBlockSwap:
    def __init__(self, type="Greedy Critical Insert"):
        self.type = type

    def apply(self, graph):
        all_solution = []
        graph.cal_sw_tw()
        cmax = graph.makespan()
        current_info = {
            'cmax': cmax,
            'solution': graph.solution,
            'source_node': None,
            'target_node': None,
        }
        node_pro_list = graph.get_critical_path()
        node_pro_machine = [graph.trans2object(node).process_machine for node in node_pro_list]
        block_seg_index = [0]
        current_machine = node_pro_machine[0]
        for i in range(len(node_pro_machine)):
            if node_pro_machine[i] != current_machine:
                block_seg_index.append(i)
                current_machine = node_pro_machine[i]
        block_seg_index.append(len(node_pro_machine))

        block_start = block_seg_index[0]
        for block_end in block_seg_index[1:]:
            block = node_pro_list[block_start:block_end]
            block_start = block_end
            if len(block) < 2:
                continue
            else:
                sn = block[0]
                stn = block[1]
                if sn[0] != stn[0]:
                    head_solution = deepcopy(graph.solution)
                    head_solution[graph.trans2object(sn).process_machine][graph.trans2object(sn).p_o] = stn
                    head_solution[graph.trans2object(stn).process_machine][graph.trans2object(stn).p_o] = sn
                    all_solution.append(head_solution)
                en = block[-1]
                etn = block[-2]
                if en[0] != etn[0]:
                    tail_solution = deepcopy(graph.solution)
                    tail_solution[graph.trans2object(en).process_machine][graph.trans2object(en).p_o] = etn
                    tail_solution[graph.trans2object(etn).process_machine][graph.trans2object(etn).p_o] = en
                    all_solution.append(tail_solution)

        return all_solution, 'BlockSwap'

    def __call__(self, *args, **kwargs):
        return self.apply(args[0])


class GreedyCriticalInsert:
    def __init__(self, type="Greedy Critical Insert"):
        self.type = type

    def apply(self, graph):
        def find_RL(s_, t_, machine):
            R = 0
            for ope in machine:
                if graph.trans2object(ope).s_w + graph.trans2object(ope).p_t <= s_:
                    R += 1
                else:
                    break
            L = 0
            for ope in machine:
                if graph.trans2object(ope).t_w + graph.trans2object(ope).p_t > t_:
                    L += 1
                else:
                    break
            return L, R

        def find_lp(pvk, s_, t_, R, L, machine):

            insert_pos = R
            min_lp = pvk + s_ + graph.trans2object(machine[R]).p_t + graph.trans2object(machine[R]).t_w
            for pos in range(R, L - 1):
                lp = pvk + graph.trans2object(machine[pos]).s_w + graph.trans2object(
                    machine[pos]).p_t + graph.trans2object(machine[pos + 1]).p_t + graph.trans2object(
                    machine[pos + 1]).t_w
                if lp < min_lp:
                    min_lp = lp
                    insert_pos = pos
            pos = L
            lp = pvk + graph.trans2object(machine[L - 1]).s_w + graph.trans2object(machine[L - 1]).p_t + t_
            if lp < min_lp:
                insert_pos = pos
            return insert_pos

        all_solution = []
        graph.cal_sw_tw()
        cmax = graph.makespan()
        current_info = {
            'cmax': cmax,
            'solution': graph.solution,
            'source_node': None,
            'target_node': None,
        }
        node_pro_list = graph.get_critical_node()
        for current_node in node_pro_list:
            # res_solution = deepcopy(graph.solution)
            # res_solution[graph.trans2object(current_node).process_machine].__delitem__(
            #     graph.trans2object(current_node).p_o)
            s_ = graph.trans2object(graph.trans2object(current_node).pre).s_w + graph.trans2object(
                graph.trans2object(current_node).pre).p_t
            t_ = graph.trans2object(graph.trans2object(current_node).sub).t_w + graph.trans2object(
                graph.trans2object(current_node).sub).p_t
            for mach_index in range(graph.machine_num):
                pvk = graph.check_process_time(current_node, mach_index)
                if pvk <= 0:
                    continue
                else:
                    res_solution = deepcopy(graph.solution)
                    res_solution[graph.trans2object(current_node).process_machine].__delitem__(
                        graph.trans2object(current_node).p_o)
                    L, R = find_RL(s_, t_, res_solution[mach_index])
                    if L <= R:
                        # insert_pos=random.sample(range(L, R+1), 1)[0]
                        insert_pos = L
                    else:
                        insert_pos = find_lp(pvk, s_, t_, R, L, graph.solution[mach_index])
                    res_solution[mach_index].insert(insert_pos, current_node)
                    all_solution.append(res_solution)

        return all_solution, 'CriticalInsert'

    def __call__(self, *args, **kwargs):
        return self.apply(args[0])


if __name__ == '__main__':
    dir_path = './dataset/1005'
    instance = 0
    for file in os.listdir(dir_path)[0:30]:
        file_path = dir_path + '/' + file
        graph = FJSPGraph(file_path)
        gantt = GanttChart(graph)
        greedy_solution = graph.generate_bottom_solution()
        graph.apply_solution(greedy_solution)
        graph.cal_sw_tw()
        cmax = graph.makespan()
        op1 = GreedyCriticalInsert()
        op2 = GreedyBlockSwap()
        op3 = GreedyCrossSwap()
        ep_cmax = []
        ep_solution = greedy_solution
        ep_min_cmax = cmax
        for i in range(40):
            result, info = random.sample([op1], 1)[0].apply(graph)
            cmax_list = []
            for solu in result:
                check = graph.apply_solution(solu)
                graph.cal_sw_tw()
                cmax = graph.makespan()
                if cmax < ep_min_cmax:
                    ep_min_cmax = cmax
                    ep_solution = solu
                cmax_list.append(cmax)
                ep_cmax.append(cmax)
                best_solution = result[np.argmin(cmax_list)]
                graph.apply_solution(best_solution)

            # print(min(cmax_list))
        # graph.apply_solution(ep_solution)
        # for machine in ep_solution:
        #     print(machine)
        # gantt.plot()
        print('Instance:', instance, '->', min(ep_cmax))
        instance += 1
