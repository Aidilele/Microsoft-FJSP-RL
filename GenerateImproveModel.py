import argparse
from utils import plot_learning_curve, create_directory
from DuelingDQN import DuelingDQN
from Enviroment import *
import torch
import time
import copy
import json
import os
import random
import time as time
import gym
import pandas as pd
import numpy as np
import pynvml
import PPO_model
from fjsp_drl_main.env.load_data import nums_detec
from EnvAgentCreate import *
import imageio
import cv2


class GIModel():
    def __init__(self, improve_env, generate_env, improve_model, generate_model):
        self.improve_env = improve_env
        self.improve_model = improve_model
        self.generate_env = generate_env
        self.generate_model = generate_model

    def schedule2solution(self, state):
        makespan = self.generate_env.schedules_batch[0]
        job_ope_index = state.opes_appertain_batch[0]
        machine_num = state.ope_ma_adj_batch.shape[-1]
        solution = []
        for i in range(machine_num):
            solution.append([])
        node_pt = {}
        adj2index = {}

        ope_index = 0
        current_job = job_ope_index[0].item()
        for i in range(len(job_ope_index)):
            if job_ope_index[i] != current_job:
                current_job = job_ope_index[i].item()
                ope_index = 0
            adj2index[i] = (current_job, ope_index)
            ope_index += 1

        for node_adj in range(makespan.shape[0]):
            node_pt[adj2index[node_adj]] = makespan[node_adj, 2].item()

        for node_adj in range(makespan.shape[0]):
            index = 0
            for node in solution[int(makespan[node_adj, 1])]:
                if node_pt[adj2index[node_adj]] > node_pt[node]:
                    index += 1
            if makespan[node_adj, 0] == 1:
                solution[int(makespan[node_adj, 1])].insert(index, adj2index[node_adj])
        return solution

    def action2solution(self, action):
        solution = deepcopy(self.improve_env.graph.solution)
        ope_adj = action[:, 0][0].item()
        mach = action[:, 0][1].item()
        solution[mach].append(self.improve_env.graph.adj2location[ope_adj])
        return solution

    def GIMschedule(self, schedule):
        state = self.generate_env.reset()
        schedule = torch.tensor(schedule)
        for action_index in range(len(schedule)):
            action = schedule[action_index].unsqueeze(-1)
            state, rewards, dones = self.generate_env.step(action)
        return state

    def generate(self, schedule):
        state = self.GIMschedule(schedule)
        with torch.no_grad():
            actions = self.generate_model.policy_old.act(state, memories=0, dones=0, flag_sample=False,
                                                         flag_train=False)
        state, rewards, dones = self.generate_env.step(actions)  # environment transit
        solution = self.schedule2solution(state)
        return solution

    def improve(self, solution, env_index=0):
        h = np.zeros((1, 8))
        c = np.zeros((1, 8))
        ope_fea, ope_adj, insert_adj, insert_mask, init_cmax = self.improve_env.eva_reset(eva_index=env_index,
                                                                                          mode='Improve',
                                                                                          solution=solution)
        best_solution = deepcopy(solution)
        best_cmax = init_cmax
        for i in range(0):
            action, hn, cn = self.improve_model.choose_action(ope_fea, ope_adj, insert_adj, insert_mask, h, c,
                                                              isTrain=True)
            next_ope_fea, next_ope_adj, next_insert_adj, next_insert_mask, cmax, done, _ = self.improve_env.step(action)
            ope_fea = next_ope_fea
            ope_adj = next_ope_adj
            insert_adj = next_insert_adj
            insert_mask = next_insert_mask
            h = hn
            c = cn
            if cmax <= best_cmax:
                best_cmax = cmax
                best_solution = self.improve_env.graph.solution
        self.improve_env.graph.apply_solution(best_solution)
        self.improve_env.graph.cal_sw_tw()
        schedule = self.improve_env.graph.trans2schedule()
        return schedule


if __name__ == '__main__':
    file_path = './dataset/1005test/'
    model_path = "./fjsp_drl_main/model/save_10_5.pt"
    generate_env, generate_agent = create_generate(file_path, model_path)
    improve_env, improve_agent = create_improve()
    for env_index in range(len(improve_env.graph_list)):
        agent = GIModel(improve_env, generate_env[env_index], improve_agent, generate_agent)
        schedule = np.array([])
        images = []
        ope_num = agent.improve_env.graph_list[env_index].ope_num
        for i in range(ope_num):
            solution = agent.generate(schedule)
            schedule = agent.improve(solution, env_index=env_index)
            # graph = copy.deepcopy(improve_env.graph_list[0])  # 此处将要画的solution复制过来
            # graph.apply_solution(solution)
            # graph.cal_sw_tw()
            # cmax = graph.makespan()
            # gantchart = GanttChart(graph)
            image = agent.improve_env.render()
            images.append(image)

        width, height = images[0].size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定编码器
        video = cv2.VideoWriter('render_result/output_' + str(env_index) + '.mp4', fourcc, 2, (width, height))
        for img in images:
            # 将 PIL 图像转换为 numpy 数组
            frame = np.array(img)
            # 将 RGB 转换为 BGR（OpenCV 使用 BGR）
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 写入视频
            video.write(frame)

        # 释放 VideoWriter 对象
        video.release()

        print(agent.improve_env.graph.makespan())
