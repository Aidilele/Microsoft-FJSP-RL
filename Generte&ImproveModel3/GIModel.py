import json
import Improve
import Generate
from torch.utils.tensorboard import SummaryWriter
from utils import create_directory
import random
import numpy as np
import copy
import time
import IEnviroment
import GEnviroment
import IDuelingDQN
import GDuelingDQN


class GIM:
    def __init__(self, config_path='./config.json'):
        with open(config_path, 'r') as load_f:
            config = json.load(load_f)
        self.config = config
        self.instance_index = 0
        self.ie = IEnviroment.FJSPEnviroment(dir_path='./dataset/' + config['CommonTrainParas']['dataset'])
        self.ge = GEnviroment.FJSPEnviroment(dir_path='./dataset/' + config['CommonTrainParas']['dataset'])
        ope_num = self.ie.graph_list[0].ope_num
        machine_num = self.ie.graph_list[0].machine_num
        job_num = self.ge.graph_list[0].job_num
        time_list = time.ctime().split(' ')
        if '' in time_list:
            time_list.remove('')
        time_clock = time_list[3].split(':')
        time_clock_str = time_clock[0] + '.' + time_clock[1] + '.' + time_clock[2]
        time_list[3] = time_clock_str
        if '' in time_list: time_list.remove('')
        time_str = time_list[1] + '_' + time_list[2] + '_' + time_list[3]
        Summer = SummaryWriter('./runs/{}_{}_{}/{}'.format(config['CommonTrainParas']['random_seed'],
                                                           config['CommonTrainParas']['reward_type'],
                                                           config['CommonTrainParas']['save_info'],
                                                           time_str))
        self.model = {
            "GI": self.GIschedule,
            "I": self.Ischedule,
            "G": self.Gschedule,
            "BI": self.BIschedule
        }
        self.summer = Summer
        self.ga = GDuelingDQN.DuelingDQN(alpha=config['GenerateTrainParas']['alpha'],
                                         state_dim=config['GenerateLoadParas']['gat_output_dim'],
                                         action_dim=config['GenerateLoadParas']['gat_output_dim'] * 3,
                                         hidden_dim=config['GenerateLoadParas']['hidden_dim'],
                                         hidden_layer=config['GenerateLoadParas']['hidden_layer_num'],
                                         ope_dim=config['GenerateLoadParas']['ope_dim'],
                                         gat_hidden_dim=config['GenerateLoadParas']['gat_hidden_dim'],
                                         gat_output_dim=config['GenerateLoadParas']['gat_output_dim'],
                                         ckpt_dir=config['GenerateLoadParas']['model_path'],
                                         ope_num=ope_num,
                                         machine_num=machine_num,
                                         gamma=config['GenerateTrainParas']['gamma'],
                                         tau=config['GenerateTrainParas']['tau'],
                                         epsilon=config['GenerateTrainParas']['epsilon'],
                                         eps_end=config['GenerateTrainParas']['eps_end'],
                                         eps_dec=config['GenerateTrainParas']['eps_dec'],
                                         max_size=config['GenerateTrainParas']['max_size'],
                                         batch_size=config['GenerateTrainParas']['batch_size'],
                                         summer=Summer)
        self.ia = IDuelingDQN.DuelingDQN(alpha=config['ImproveTrainParas']['alpha'],
                                         state_dim=config['ImproveLoadParas']['gat_output_dim'],
                                         action_dim=config['ImproveLoadParas']['gat_output_dim'] * 2,
                                         hidden_dim=config['ImproveLoadParas']['hidden_dim'],
                                         hidden_layer=config['ImproveLoadParas']['hidden_layer_num'],
                                         ope_dim=config['ImproveLoadParas']['ope_dim'],
                                         gat_hidden_dim=config['ImproveLoadParas']['gat_hidden_dim'],
                                         gat_output_dim=config['ImproveLoadParas']['gat_output_dim'],
                                         ckpt_dir=config['ImproveLoadParas']['model_path'],
                                         ope_num=ope_num,
                                         machine_num=machine_num,
                                         gamma=config['ImproveTrainParas']['gamma'],
                                         tau=config['ImproveTrainParas']['tau'],
                                         epsilon=config['ImproveTrainParas']['epsilon'],
                                         eps_end=config['ImproveTrainParas']['eps_end'],
                                         eps_dec=config['ImproveTrainParas']['eps_dec'],
                                         max_size=config['ImproveTrainParas']['max_size'],
                                         batch_size=config['ImproveTrainParas']['batch_size'],
                                         summer=Summer)
        insert_num = ope_num + machine_num
        self.improve_ep_buffer = [
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], ope_num + 2, config['ImproveLoadParas']['ope_dim']]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], ope_num + 2, ope_num + 2]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], insert_num, ope_num + 2]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], ope_num * insert_num]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], config['ImproveLoadParas']['gat_output_dim']]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], config['ImproveLoadParas']['gat_output_dim']]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"]]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"]]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], ope_num + 2, config['ImproveLoadParas']['ope_dim']]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], ope_num + 2, ope_num + 2]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], insert_num, ope_num + 2]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], ope_num * insert_num]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], config['ImproveLoadParas']['gat_output_dim']]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"], config['ImproveLoadParas']['gat_output_dim']]),
            np.zeros([config["ImproveTrainParas"]["ep_max_step"]]),
        ]
        self.generate_ep_buffer = [
            np.zeros([ope_num, ope_num + 2, config['GenerateLoadParas']['ope_dim']]),
            np.zeros([ope_num, ope_num + 2, ope_num + 2]),
            np.zeros([ope_num, job_num, ope_num + 2]),
            np.zeros([ope_num, machine_num, ope_num + 2]),
            np.zeros([ope_num, job_num * machine_num]),
            np.zeros([ope_num, job_num]),
            np.zeros([ope_num, config['GenerateLoadParas']['gat_output_dim']]),
            np.zeros([ope_num, config['GenerateLoadParas']['gat_output_dim']]),
            np.zeros([ope_num]),
            np.zeros([ope_num + 1]),
            np.zeros([ope_num, ope_num + 2, config['GenerateLoadParas']['ope_dim']]),
            np.zeros([ope_num, ope_num + 2, ope_num + 2]),
            np.zeros([ope_num, job_num, ope_num + 2]),
            np.zeros([ope_num, machine_num, ope_num + 2]),
            np.zeros([ope_num, job_num * machine_num]),
            np.zeros([ope_num, job_num]),
            np.zeros([ope_num, config['GenerateLoadParas']['gat_output_dim']]),
            np.zeros([ope_num, config['GenerateLoadParas']['gat_output_dim']]),
            np.zeros([ope_num]),
        ]

    def reset(self):
        self.ins_index = self.instance_selector(self.config["CommonTrainParas"]["init_mode"])
        self.ge.reset(ins_index=self.ins_index, solution=0)
        self.ie.reset(ins_index=self.ins_index)
        self.solution = 0

    def instance_selector(self, mode="Random"):
        if mode == 'Random':
            return random.randint(0, len(self.ie.graph_list))
        elif mode == 'Circulate':
            index = self.instance_index
            self.instance_index = (self.instance_index + 1) % len(self.ie.graph_list)
            return index

    def generate(self, solution=None, step=0):
        state, done = self.ge.reset(ins_index=self.ins_index, solution=solution)
        if step == 0:
            h = np.zeros((1, self.config['GenerateLoadParas']['gat_output_dim']))
            c = np.zeros((1, self.config['GenerateLoadParas']['gat_output_dim']))
        else:
            h = self.h
            c = self.c
        self.generate_ep_buffer[0][step] = state[0]
        self.generate_ep_buffer[1][step] = state[1]
        self.generate_ep_buffer[2][step] = state[2]
        self.generate_ep_buffer[3][step] = state[3]
        self.generate_ep_buffer[4][step] = state[4]
        self.generate_ep_buffer[5][step] = state[6]
        self.generate_ep_buffer[6][step] = h
        self.generate_ep_buffer[7][step] = c
        self.generate_ep_buffer[9][step] = state[5]
        if step > 0:
            self.generate_ep_buffer[10][step - 1] = state[0]
            self.generate_ep_buffer[11][step - 1] = state[1]
            self.generate_ep_buffer[12][step - 1] = state[2]
            self.generate_ep_buffer[13][step - 1] = state[3]
            self.generate_ep_buffer[14][step - 1] = state[4]
            self.generate_ep_buffer[15][step - 1] = state[6]
            self.generate_ep_buffer[16][step - 1] = h
            self.generate_ep_buffer[17][step - 1] = c
            self.generate_ep_buffer[18][step - 1] = False
        action, hn, cn = self.ga.choose_action(state, h, c, isTrain=True)
        self.h = hn
        self.c = cn
        n_state, done, _ = self.ge.step(action)
        self.generate_ep_buffer[8][step] = action
        self.solution = self.ge.graph.solution
        return self.ge.graph.solution

    def store_generate(self):
        rewards = self.ge.reward(self.config['GenerateTrainParas']['reward_type'], self.generate_ep_buffer[9])
        ep_buffer_size = len(rewards)
        self.generate_ep_buffer[9][:ep_buffer_size] = rewards
        self.ga.memory.store(
            self.generate_ep_buffer[0][:ep_buffer_size],
            self.generate_ep_buffer[1][:ep_buffer_size],
            self.generate_ep_buffer[2][:ep_buffer_size],
            self.generate_ep_buffer[3][:ep_buffer_size],
            self.generate_ep_buffer[4][:ep_buffer_size],
            self.generate_ep_buffer[5][:ep_buffer_size],
            self.generate_ep_buffer[6][:ep_buffer_size],
            self.generate_ep_buffer[7][:ep_buffer_size],
            self.generate_ep_buffer[8][:ep_buffer_size],
            self.generate_ep_buffer[9][:ep_buffer_size],
            self.generate_ep_buffer[10][:ep_buffer_size],
            self.generate_ep_buffer[11][:ep_buffer_size],
            self.generate_ep_buffer[12][:ep_buffer_size],
            self.generate_ep_buffer[13][:ep_buffer_size],
            self.generate_ep_buffer[14][:ep_buffer_size],
            self.generate_ep_buffer[15][:ep_buffer_size],
            self.generate_ep_buffer[16][:ep_buffer_size],
            self.generate_ep_buffer[17][:ep_buffer_size],
            self.generate_ep_buffer[18][:ep_buffer_size],
            ep_buffer_size,
        )

    def cal_improve_step(self, index):
        # improve_step = 0
        # if step > 25:
        improve_step = max(1, int(index / 10))
        return improve_step

    def eva_improve(self, solution=0, index=5):
        reward_info = {}
        ope_fea, ope_adj, insert_adj, insert_mask, init_cmax = self.ie.reset(self.ins_index, solution)
        reward_info['init_cmax'] = init_cmax
        cmax_buffer = [init_cmax]
        solution_buffer = [self.ie.graph.solution]
        h = np.zeros((1, self.config['ImproveLoadParas']['gat_output_dim']))
        c = np.zeros((1, self.config['ImproveLoadParas']['gat_output_dim']))
        for step in range(index):
            action, hn, cn = self.ia.choose_action(ope_fea, ope_adj, insert_adj, insert_mask, h, c, isTrain=True)
            next_ope_fea, next_ope_adj, next_insert_adj, next_insert_mask, cmax, done, _ = self.ie.step(action)
            cmax_buffer.append(cmax)
            solution_buffer.append(copy.deepcopy(self.ie.graph.solution))
            ope_fea = next_ope_fea
            ope_adj = next_ope_adj
            insert_adj = next_insert_adj
            insert_mask = next_insert_mask
            h = hn
            c = cn
        self.solution = solution_buffer[-1]
        return (solution_buffer[-1], solution_buffer[np.argmin(cmax_buffer)], np.min(cmax_buffer))

    def improve(self, solution=0, index=5):
        reward_info = {}
        ope_fea, ope_adj, insert_adj, insert_mask, init_cmax = self.ie.reset(self.ins_index, solution)
        reward_info['init_cmax'] = init_cmax
        cmax_buffer = [init_cmax]
        solution_buffer = [self.ie.graph.solution]
        h = np.zeros((1, self.config['ImproveLoadParas']['gat_output_dim']))
        c = np.zeros((1, self.config['ImproveLoadParas']['gat_output_dim']))
        for step in range(index):
            self.improve_ep_buffer[0][step] = ope_fea
            self.improve_ep_buffer[1][step] = ope_adj
            self.improve_ep_buffer[2][step] = insert_adj
            self.improve_ep_buffer[3][step] = insert_mask
            self.improve_ep_buffer[4][step] = h
            self.improve_ep_buffer[5][step] = c
            action, hn, cn = self.ia.choose_action(ope_fea, ope_adj, insert_adj, insert_mask, h, c, isTrain=True)
            next_ope_fea, next_ope_adj, next_insert_adj, next_insert_mask, cmax, done, _ = self.ie.step(action)
            self.improve_ep_buffer[8][step] = next_ope_fea
            self.improve_ep_buffer[9][step] = next_ope_adj
            self.improve_ep_buffer[10][step] = next_insert_adj
            self.improve_ep_buffer[11][step] = next_insert_mask
            self.improve_ep_buffer[12][step] = hn
            self.improve_ep_buffer[13][step] = cn
            self.improve_ep_buffer[14][step] = done
            self.improve_ep_buffer[6][step] = action
            cmax_buffer.append(cmax)
            solution_buffer.append(copy.deepcopy(self.ie.graph.solution))
            ope_fea = next_ope_fea
            ope_adj = next_ope_adj
            insert_adj = next_insert_adj
            insert_mask = next_insert_mask
            h = hn
            c = cn
        rewards = self.ie.reward(self.config['ImproveTrainParas']['reward_type'], cmax_buffer, reward_info)
        ep_buffer_size = len(rewards)
        self.improve_ep_buffer[7][:ep_buffer_size] = rewards
        self.ia.memory.store(self.improve_ep_buffer[0][:ep_buffer_size],
                             self.improve_ep_buffer[1][:ep_buffer_size],
                             self.improve_ep_buffer[2][:ep_buffer_size],
                             self.improve_ep_buffer[3][:ep_buffer_size],
                             self.improve_ep_buffer[4][:ep_buffer_size],
                             self.improve_ep_buffer[5][:ep_buffer_size],
                             self.improve_ep_buffer[6][:ep_buffer_size],
                             self.improve_ep_buffer[7][:ep_buffer_size],
                             self.improve_ep_buffer[8][:ep_buffer_size],
                             self.improve_ep_buffer[9][:ep_buffer_size],
                             self.improve_ep_buffer[10][:ep_buffer_size],
                             self.improve_ep_buffer[11][:ep_buffer_size],
                             self.improve_ep_buffer[12][:ep_buffer_size],
                             self.improve_ep_buffer[13][:ep_buffer_size],
                             self.improve_ep_buffer[14][:ep_buffer_size],
                             ep_buffer_size,
                             )
        self.solution = solution_buffer[-1]
        return (solution_buffer[-1], solution_buffer[np.argmin(cmax_buffer)], np.min(cmax_buffer))

    def update(self, repeat=5, model='All'):
        if model == 'Improve' or model == 'All':
            if self.ia.memory.total_count > self.ia.memory.max_size:
                for i in range(repeat):
                    self.ia.learn()
        if model == 'Generate' or model == 'All':
            if self.ga.memory.total_count > self.ga.memory.max_size:
                for i in range(repeat):
                    self.ga.learn()

    def GIschedule(self):
        self.reset()
        solution = 0
        for i in range(self.ge.graph.ope_num):
            solution = self.generate(solution, i)
            if i > (self.ge.graph.ope_num - self.config['ImproveTrainParas']['ep_max_step'] // 10):
                solution = self.improve(solution, 10)
        n_state, _ = self.ge.reset(self.ins_index, solution)
        self.generate_ep_buffer[9][-1] = n_state[5]
        self.generate_ep_buffer[10][-1] = n_state[0]
        self.generate_ep_buffer[11][-1] = n_state[1]
        self.generate_ep_buffer[12][-1] = n_state[2]
        self.generate_ep_buffer[13][-1] = n_state[3]
        self.generate_ep_buffer[14][-1] = n_state[4]
        self.generate_ep_buffer[15][-1] = n_state[6]
        self.generate_ep_buffer[16][-1] = self.h
        self.generate_ep_buffer[17][-1] = self.c
        self.generate_ep_buffer[18][-1] = True
        self.store_generate()
        return n_state[5]

    def BIschedule(self, eva_flag=False):
        self.reset()
        solution = 0
        if eva_flag:
            for i in range(self.ge.graph.ope_num):
                solution = self.ge.bottomlevel_generate(solution)
                if i > (self.ge.graph.ope_num - self.config['ImproveTrainParas']['ep_max_step'] // 10):
                    solution, _, _ = self.eva_improve(solution, 5)
            _, solution, best_cmax = self.eva_improve(solution, self.config['ImproveTrainParas']['ep_max_step'] // 2)
        else:
            for i in range(self.ge.graph.ope_num):
                solution = self.ge.bottomlevel_generate(solution)
                if i > (self.ge.graph.ope_num - self.config['ImproveTrainParas']['ep_max_step'] // 10):
                    solution, _, _ = self.improve(solution, 5)
            _, solution, best_cmax = self.improve(solution, self.config['ImproveTrainParas']['ep_max_step'] // 2)
        print(best_cmax)
        return best_cmax

    def Gschedule(self):
        self.reset()
        solution = 0
        for i in range(self.ge.graph.ope_num):
            solution = self.generate(solution, i)
        n_state, _ = self.ge.reset(self.ins_index, solution)
        self.generate_ep_buffer[9][-1] = n_state[5]
        self.generate_ep_buffer[10][-1] = n_state[0]
        self.generate_ep_buffer[11][-1] = n_state[1]
        self.generate_ep_buffer[12][-1] = n_state[2]
        self.generate_ep_buffer[13][-1] = n_state[3]
        self.generate_ep_buffer[14][-1] = n_state[4]
        self.generate_ep_buffer[15][-1] = n_state[6]
        self.generate_ep_buffer[16][-1] = self.h
        self.generate_ep_buffer[17][-1] = self.c
        self.generate_ep_buffer[18][-1] = True
        self.store_generate()
        return n_state[5]

    def Ischedule(self):
        self.reset()
        solution = self.improve(index=self.config['ImproveTrainParas']['ep_max_step'])
        try:
            self.ie.graph.apply_solution(solution)
        except:
            print(0)
        self.ie.graph.cal_sw_tw()
        init_cmax = self.ie.graph.makespan()
        return init_cmax

    def load(self):
        generate_model_info = {}
        generate_model_info['seed'] = self.config["CommonLoadParas"]["load_seed"]
        generate_model_info['reward_type'] = self.config["GenerateLoadParas"]["load_reward_type"]
        generate_model_info['episode'] = self.config["GenerateLoadParas"]["load_ep"]
        generate_model_info['info'] = self.config["GenerateLoadParas"]["load_info"]
        generate_model_info['model_structure'] = "{}_{}_{}_{}_{}".format(self.config['GenerateLoadParas']['ope_dim'],
                                                                         self.config['GenerateLoadParas'][
                                                                             'gat_hidden_dim'],
                                                                         self.config['GenerateLoadParas'][
                                                                             'gat_output_dim'],
                                                                         self.config['GenerateLoadParas']['hidden_dim'],
                                                                         self.config['GenerateLoadParas'][
                                                                             'hidden_layer_num'])
        improve_model_info = {}
        improve_model_info['seed'] = self.config["CommonLoadParas"]["load_seed"]
        improve_model_info['reward_type'] = self.config["ImproveLoadParas"]["load_reward_type"]
        improve_model_info['episode'] = self.config["ImproveLoadParas"]["load_ep"]
        improve_model_info['info'] = self.config["ImproveLoadParas"]["load_info"]
        improve_model_info['model_structure'] = "{}_{}_{}_{}_{}".format(self.config['ImproveLoadParas']['ope_dim'],
                                                                        self.config['ImproveLoadParas'][
                                                                            'gat_hidden_dim'],
                                                                        self.config['ImproveLoadParas'][
                                                                            'gat_output_dim'],
                                                                        self.config['ImproveLoadParas']['hidden_dim'],
                                                                        self.config['ImproveLoadParas'][
                                                                            'hidden_layer_num'])
        if self.config["ImproveLoadParas"]["load_ep"]:
            self.ia.load_models(improve_model_info)
        if self.config["GenerateLoadParas"]["load_ep"]:
            self.ga.load_models(generate_model_info)
        create_directory(self.config['ImproveLoadParas']['model_path'], sub_dirs=['Q_eval', 'Q_target'])
        create_directory(self.config['GenerateLoadParas']['model_path'], sub_dirs=['Q_eval', 'Q_target'])

    def save(self, episode):
        generate_model_info = {}
        generate_model_info['episode'] = episode
        generate_model_info['seed'] = self.config["CommonTrainParas"]["random_seed"]
        generate_model_info['reward_type'] = self.config["GenerateTrainParas"]["reward_type"]
        generate_model_info['info'] = self.config["CommonTrainParas"]["save_info"]
        generate_model_info['mode_structure'] = "{}_{}_{}_{}_{}".format(self.config['GenerateLoadParas']['ope_dim'],
                                                                        self.config['GenerateLoadParas'][
                                                                            'gat_hidden_dim'],
                                                                        self.config['GenerateLoadParas'][
                                                                            'gat_output_dim'],
                                                                        self.config['GenerateLoadParas']['hidden_dim'],
                                                                        self.config['GenerateLoadParas'][
                                                                            'hidden_layer_num'])
        improve_model_info = {}
        improve_model_info['episode'] = episode
        improve_model_info['seed'] = self.config["CommonTrainParas"]["random_seed"]
        improve_model_info['reward_type'] = self.config["ImproveTrainParas"]["reward_type"]
        improve_model_info['info'] = self.config["CommonTrainParas"]["save_info"]
        improve_model_info['mode_structure'] = "{}_{}_{}_{}_{}".format(self.config['ImproveLoadParas']['ope_dim'],
                                                                       self.config['ImproveLoadParas'][
                                                                           'gat_hidden_dim'],
                                                                       self.config['ImproveLoadParas'][
                                                                           'gat_output_dim'],
                                                                       self.config['ImproveLoadParas']['hidden_dim'],
                                                                       self.config['ImproveLoadParas'][
                                                                           'hidden_layer_num'])
        self.ia.save_models(improve_model_info)
        self.ga.save_models(generate_model_info)

    def print_info(self, info):
        return 0

    def other(self):
        return 0
