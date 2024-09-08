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
from fjsp_drl_main.env.fjsp_env import FJSPEnv


def create_improve():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)

    train_paras = load_dict['TrainParas']
    load_model_paras = load_dict['LoadModelParas']

    # 加载本次训练参数
    max_episodes = train_paras['max_episodes']
    ep_max_step = train_paras['ep_max_step']
    random_seed = train_paras['random_seed']
    alpha = train_paras['alpha']
    gamma = train_paras['gamma']
    tau = train_paras['tau']
    epsilon = train_paras['epsilon']
    eps_end = train_paras['eps_end']
    eps_dec = train_paras['eps_dec']
    max_size = train_paras['max_size']
    batch_size = train_paras['batch_size']
    reward_type = train_paras['reward_type']
    dataset = train_paras['dataset']
    reset_mode = train_paras['reset_mode']
    render = train_paras['render']
    save_info = train_paras['save_info']
    save_model_info = {}
    save_model_info['seed'] = random_seed
    save_model_info['reward_type'] = reward_type
    save_model_info['info'] = save_info

    # 加载预训练模型参数
    ope_dim = load_model_paras['ope_dim']
    gat_hidden_dim = load_model_paras['gat_hidden_dim']
    gat_output_dim = load_model_paras['gat_output_dim']
    state_dim = gat_output_dim
    action_dim = gat_output_dim * 2
    hidden_dim = load_model_paras['hidden_dim']
    hidden_layer_num = load_model_paras['hidden_layer_num']
    load_reward_type = load_model_paras['load_reward_type']
    load_seed = load_model_paras['load_seed']
    load_ep = load_model_paras['load_ep']
    load_info = load_model_paras['load_info']
    load_model_info = {}
    load_model_info['seed'] = load_seed
    load_model_info['reward_type'] = load_reward_type
    load_model_info['episode'] = load_ep
    load_model_info['info'] = load_info
    load_model_info['model_structure'] = "{}_{}_{}_{}_{}".format(ope_dim, gat_hidden_dim, gat_output_dim, hidden_dim,
                                                                 hidden_layer_num)

    # random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

    env = FJSPEnviroment(dir_path='./dataset/' + dataset, env_tpye='Evaluate')
    env.reset()
    ope_num = env.graph.ope_num
    machine_num = env.graph.machine_num
    agent = DuelingDQN(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                       hidden_dim=hidden_dim, hidden_layer=hidden_layer_num,
                       ope_dim=ope_dim, gat_hidden_dim=gat_hidden_dim,
                       gat_output_dim=gat_output_dim, ckpt_dir='./checkpoints/DuelingDQN/', ope_num=ope_num,
                       machine_num=machine_num,
                       gamma=gamma, tau=tau, epsilon=0.0,
                       eps_end=0.0, eps_dec=0.0, max_size=max_size, batch_size=batch_size)
    if load_ep:
        agent.load_models(load_model_info)
    return env, agent


def create_generate(file_path='./dataset/1005', model_path="./fjsp_drl_main/model/save_10_5.pt"):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None,
                           sci_mode=False)
    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    test_paras = load_dict["test_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    num_ins = test_paras["num_ins"]
    if test_paras["sample"]:
        env_test_paras["batch_size"] = test_paras["num_sample"]
    else:
        env_test_paras["batch_size"] = 1
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    # data_path = "./data_test/{0}/".format(test_paras["data_path"])
    data_path = "./{0}".format(file_path[2:]) + '/'
    test_files = os.listdir(data_path)
    test_files.sort()
    test_files = test_files[:num_ins]
    model = PPO_model.PPO(model_paras, train_paras)
    envs = []  # Store multiple environments
    model_CKPT = torch.load('./fjsp_drl_main/results/save_20_10.pt')
    model.policy.load_state_dict(model_CKPT)
    model.policy_old.load_state_dict(model_CKPT)
    for i_ins in range(len(test_files)):
        test_file = data_path + test_files[i_ins]
        with open(test_file) as file_object:
            line = file_object.readlines()
            ins_num_jobs, ins_num_mas, _ = nums_detec(line)
        env_test_paras["num_jobs"] = ins_num_jobs
        env_test_paras["num_mas"] = ins_num_mas
        # Environment object already exists
        # Clear the existing environment
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # if meminfo.used / meminfo.total > 0.7:
        #     envs.clear()
        env = FJSPEnv( case=[test_file], env_paras=env_test_paras, data_source='file')
        envs.append(copy.deepcopy(env))
            # print("Create env[{0}]".format(i_ins))

        # Schedule an instance/environment
        # DRL-S
    return envs, model
