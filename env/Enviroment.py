from env.Operator import *
import test as drl
from env.GanttChart import GanttChart


class FJSPEnviroment():

    def __init__(self, dir_path='./dataset/1005', env_tpye="NoDrl"):
        self.graph_list = []
        self.state = 0
        self.reward_history = []
        self.cmax_min = float('Inf')
        self.instance = os.listdir(dir_path)
        self.instance.sort()
        self.operator_list = [GreedyCriticalInsert(), GreedyBlockSwap(), GreedyCrossSwap()]
        self.instance_index = 0
        for file in self.instance:
            file_path = dir_path + '/' + file
            graph = FJSPGraph(file_path)
            self.graph_list.append(graph)
        if env_tpye == 'Train':
            self.drl_solution = drl.main()
        self.dir_path = dir_path

    def get_state(self):
        ope_fea, ope_adj, insert_adj = self.graph.get_fea()
        insert_mask = self.graph.get_insert_mask()

        return ope_fea, ope_adj, insert_adj, insert_mask

    def reset(self, mode='BottomLevel', init_mode='Random', solution=None):
        if init_mode == 'Random':
            self.instance_index = random.randint(0, len(self.graph_list))
            self.graph = self.graph_list[self.instance_index]
        elif init_mode == 'Recurrent':
            self.graph = self.graph_list[self.instance_index]
        self.graph.reset()
        if mode == 'Greedy':
            self.graph.apply_solution(self.graph.generate_greedy_solution())
        elif mode == 'Random':
            self.graph.apply_solution(self.graph.generate_random_solution())
        elif mode == 'BottomLevel':
            self.graph.apply_solution(self.graph.generate_bottom_solution())
        elif mode == 'Improve':
            self.graph.apply_solution(solution)
        elif mode == 'Drl':
            self.graph.apply_solution(self.drl_solution[self.instance_index])
        self.instance_index = (self.instance_index + 1) % len(self.graph_list)
        self.graph.cal_sw_tw()
        init_cmax = self.graph.makespan()
        ope_fea, ope_adj, insert_adj, insert_mask = self.get_state()
        return ope_fea, ope_adj, insert_adj, insert_mask, init_cmax

    def eva_reset(self, eva_index, mode='BottomLevel', solution=None):
        self.graph = self.graph_list[eva_index]
        self.graph.reset()
        if mode == 'Greedy':
            self.graph.apply_solution(self.graph.generate_greedy_solution())
        elif mode == 'Random':
            self.graph.apply_solution(self.graph.generate_random_solution())
        elif mode == 'BottomLevel':
            self.graph.apply_solution(self.graph.generate_bottom_solution())
        elif mode == 'Improve':
            self.graph.apply_solution(solution)
        elif mode == 'Drl':
            self.graph.apply_solution(drl.main2(self.dir_path, eva_index))
        self.graph.cal_sw_tw()
        init_cmax = self.graph.makespan()
        ope_fea, ope_adj, insert_adj, insert_mask = self.get_state()
        return ope_fea, ope_adj, insert_adj, insert_mask, init_cmax

    def sample(self, action_list):
        return action_list[random.sample(range(len(action_list)), 1)[0]].apply(self.graph)

    def step(self, action):
        cmax = self.graph.excute_action(action)
        ope_fea, ope_adj, insert_adj, insert_mask = self.get_state()
        return ope_fea, ope_adj, insert_adj, insert_mask, cmax, False, None

    def render(self):
        ganttchart = GanttChart(self.graph)
        image = ganttchart.plot()
        return image

    def reward(self, reward_type, buffer, info=None):
        if reward_type == 1:
            reward = self.reward1(buffer, info)
        elif reward_type == 2:
            reward = self.reward2(buffer, info)
        elif reward_type == 3:
            reward = self.reward3(buffer, info)
        elif reward_type == 4:
            reward = self.reward4(buffer, info)
        elif reward_type == 5:
            reward = self.reward5(buffer, info)
        else:
            reward = self.reward1(buffer, info)
        return reward

    def reward1(self, buffer, info=None):  # 不计算尾端奖励，分两种情况计算奖励，奖励只会为全正或全负
        seg = [0]
        init_cmax = buffer[0]
        min_cmax = init_cmax
        buffer = buffer[1:]
        reward = []
        for index in range(len(buffer) - 1):
            if buffer[index] < min_cmax:
                min_cmax = buffer[index]
                if buffer[index + 1] >= buffer[index]:
                    seg.append(index + 1)
        last_sub = [init_cmax]
        for index in range(len(seg) - 1):
            sub_buffer = buffer[seg[index]:seg[index + 1]]
            local = (last_sub[-1] - sub_buffer[-1]) / len(sub_buffer)
            for cmax in sub_buffer:
                glob = np.clip(last_sub[-1] - cmax, -0.5 * np.abs(local), 0.5 * np.abs(local))
                exp_reward = local + glob
                reward.append(exp_reward)
            last_sub = sub_buffer

        if len(reward) == 0:
            seg = [0]
            init_cmax = buffer[0]
            max_cmax = init_cmax
            buffer = buffer[1:]
            reward = []
            for index in range(len(buffer) - 1):
                if buffer[index] > max_cmax:
                    max_cmax = buffer[index]
                    if buffer[index + 1] <= buffer[index]:
                        seg.append(index + 1)
            last_sub = [init_cmax]
            for index in range(len(seg) - 1):
                sub_buffer = buffer[seg[index]:seg[index + 1]]
                local = (last_sub[-1] - sub_buffer[-1]) / len(sub_buffer)
                for cmax in sub_buffer:
                    glob = np.clip(buffer[0] - cmax, -0.5 * np.abs(local), 0.5 * np.abs(local))
                    exp_reward = local + glob
                    reward.append(exp_reward)
                last_sub = sub_buffer

        # if len(seg) == 1:
        #     last_index = 0
        # else:
        #     last_index = seg[index + 1]
        # local = (buffer[last_index] - buffer[-1]) / (len(buffer) - last_index)
        # for cmax in buffer[last_index:]:
        #     glob = buffer[0] - cmax
        #     exp_reward = local + glob
        #     reward.append(exp_reward)
        return reward

    def reward4(self, buffer, info=None):  # 不计算尾端奖励,累计Reward等于最终的Cmax decady
        seg = [0]
        init_cmax = buffer[0]
        min_cmax = init_cmax
        buffer = buffer[1:]
        reward = []
        for index in range(len(buffer) - 1):
            if buffer[index] < min_cmax:
                min_cmax = buffer[index]
                if buffer[index + 1] >= buffer[index]:
                    seg.append(index + 1)
        last_sub = [init_cmax]
        for index in range(len(seg) - 1):
            sub_buffer = buffer[seg[index]:seg[index + 1]]
            local = (last_sub[-1] - sub_buffer[-1]) / len(sub_buffer)
            for cmax in sub_buffer:
                exp_reward = local
                reward.append(exp_reward)
            last_sub = sub_buffer
        # if len(seg) == 1:
        #     last_index = 0
        # else:
        #     last_index = seg[index + 1]
        # local = (buffer[last_index] - buffer[-1]) / (len(buffer) - last_index)
        # for cmax in buffer[last_index:]:
        #     glob = buffer[0] - cmax
        #     exp_reward = local + glob
        #     reward.append(exp_reward)
        return reward

    #
    def reward2(self, buffer, info=None):  # 在reward1的基础上，计算尾端奖励，奖励会出现正负都有
        seg = [0]
        init_cmax = buffer[0]
        min_cmax = init_cmax
        buffer = buffer[1:]
        reward = []
        for index in range(len(buffer) - 1):
            if buffer[index] < min_cmax:
                min_cmax = buffer[index]
                if buffer[index + 1] >= buffer[index]:
                    seg.append(index + 1)
        last_sub = [init_cmax]
        for index in range(len(seg) - 1):
            sub_buffer = buffer[seg[index]:seg[index + 1]]
            local = (last_sub[-1] - sub_buffer[-1]) / len(sub_buffer)
            for cmax in sub_buffer:
                glob = np.clip(buffer[0] - cmax, -0.5 * np.abs(local), 0.5 * np.abs(local))
                exp_reward = local + glob
                reward.append(exp_reward)
            last_sub = sub_buffer

        if len(seg) == 1:
            last_index = 0
        else:
            last_index = seg[index + 1]
        sub_buffer = buffer[last_index:]
        local = (buffer[last_index] - sub_buffer[-1]) / len(sub_buffer)
        for cmax in sub_buffer:
            glob = np.clip(buffer[0] - cmax, -0.5 * np.abs(local), 0.5 * np.abs(local))
            exp_reward = local + glob
            reward.append(exp_reward)
        return reward

    def reward3(self, buffer, info=None):  # 在reward2的基础上使用相邻两步的cmax差值作为glob奖励
        seg = [0]
        init_cmax = buffer[0]
        min_cmax = init_cmax
        buffer = buffer[1:]
        reward = []
        for index in range(len(buffer) - 1):
            if buffer[index] < min_cmax:
                min_cmax = buffer[index]
                if buffer[index + 1] >= buffer[index]:
                    seg.append(index + 1)
        last_sub = [init_cmax]
        for index in range(len(seg) - 1):
            sub_buffer = buffer[seg[index]:seg[index + 1]]
            local = (last_sub[-1] - sub_buffer[-1]) / len(sub_buffer)
            last_cmax = last_sub[-1]
            for cmax in sub_buffer:
                glob = np.clip(last_cmax - cmax, -0.5 * np.abs(local), 0.5 * np.abs(local))
                exp_reward = local + glob
                reward.append(exp_reward)
                last_cmax = cmax
            last_sub = sub_buffer

        if len(seg) == 1:
            last_index = 0
        else:
            last_index = seg[index + 1]
        sub_buffer = buffer[last_index:]
        local = (buffer[last_index] - sub_buffer[-1]) / len(sub_buffer)
        last_cmax = last_sub[-1]
        for cmax in sub_buffer:
            glob = np.clip(last_cmax - cmax, -0.5 * np.abs(local), 0.5 * np.abs(local))
            exp_reward = local + glob
            reward.append(exp_reward)
            last_cmax = cmax

        return reward

    def reward5(self, buffer, info=None):  # 在reward2的基础上使用相邻两步的cmax差值作为glob奖励
        seg = [0]
        init_cmax = buffer[0]
        min_cmax = init_cmax
        buffer = buffer[1:]
        reward = []
        for index in range(len(buffer) - 1):
            if buffer[index] < min_cmax:
                min_cmax = buffer[index]
                if buffer[index + 1] >= buffer[index]:
                    seg.append(index + 1)
        last_sub = [init_cmax]
        for index in range(len(seg) - 1):
            sub_buffer = buffer[seg[index]:seg[index + 1]]
            local = (last_sub[-1] / sub_buffer[-1]) - 1
            last_cmax = last_sub[-1]
            for cmax in sub_buffer:
                glob = np.clip(last_cmax / cmax - 1, -0.5 * np.abs(local), 0.5 * np.abs(local))
                exp_reward = local + glob
                reward.append(exp_reward)
                last_cmax = cmax
            last_sub = sub_buffer

        if len(seg) == 1:
            last_index = 0
        else:
            last_index = seg[index + 1]
        sub_buffer = buffer[last_index:]
        local = (buffer[last_index] / sub_buffer[-1]) - 1
        last_cmax = last_sub[-1]
        for cmax in sub_buffer:
            glob = np.clip(last_cmax / cmax - 1, -0.5 * np.abs(local), 0.5 * np.abs(local))
            exp_reward = local + glob
            reward.append(exp_reward)
            last_cmax = cmax
        return reward
