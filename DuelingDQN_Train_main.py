import argparse
from utils.utils import create_directory
from model.DuelingDQN import DuelingDQN
from env.Enviroment import *
from torch.utils.tensorboard import SummaryWriter
import json
import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DuelingDQN/')
parser.add_argument('--reward_path', type=str, default='./output_images/reward.png')
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')
args = parser.parse_args()


def main():
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)

    train_paras = load_dict['TrainParas']
    load_model_paras = load_dict['LoadModelParas']

    # load training parameters
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
    save_info = train_paras['save_info']

    save_model_info = {}
    save_model_info['seed'] = random_seed
    save_model_info['reward_type'] = reward_type
    save_model_info['info'] = save_info

    # load pretrain parameters
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
    save_model_info['mode_structure'] = "{}_{}_{}_{}_{}".format(ope_dim, gat_hidden_dim, gat_output_dim, hidden_dim,
                                                                hidden_layer_num)

    time_list = time.ctime().split(' ')
    if '' in time_list:
        time_list.remove('')
    time_clock = time_list[3].split(':')
    time_clock_str = time_clock[0] + '.' + time_clock[1] + '.' + time_clock[2]
    time_list[3] = time_clock_str
    if '' in time_list: time_list.remove('')
    # time_str = ''
    time_str = time_list[1] + '_' + time_list[2] + '_' + time_list[3]
    save_model_info['time'] = time_str

    # random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

    env = FJSPEnviroment(dir_path='./dataset/' + dataset)
    env.reset()
    ope_num = env.graph.ope_num
    machine_num = env.graph.machine_num
    Summer = SummaryWriter('./runs/{}_{}_{}/{}'.format(save_model_info['seed'],
                                                       save_model_info['reward_type'],
                                                       save_model_info['info'],
                                                       save_model_info['time']))

    agent = DuelingDQN(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                       hidden_dim=hidden_dim, hidden_layer=hidden_layer_num,
                       ope_dim=ope_dim, gat_hidden_dim=gat_hidden_dim,
                       gat_output_dim=gat_output_dim, ckpt_dir=args.ckpt_dir, ope_num=ope_num, machine_num=machine_num,
                       gamma=gamma, tau=tau, epsilon=epsilon,
                       eps_end=eps_end, eps_dec=eps_dec, max_size=max_size, batch_size=batch_size,
                       summer=Summer)
    if load_ep:
        agent.load_models(load_model_info)
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    reward_info = {}

    insert_num = env.graph.ope_num + env.graph.machine_num
    ep_ope_fea = np.zeros([ep_max_step, ope_num + 2, ope_dim])
    ep_ope_adj = np.zeros([ep_max_step, ope_num + 2, ope_num + 2])
    ep_insert_adj = np.zeros([ep_max_step, insert_num, ope_num + 2])
    ep_insert_mask = np.zeros([ep_max_step, ope_num * insert_num])
    ep_h = np.zeros([ep_max_step, gat_output_dim])
    ep_c = np.zeros([ep_max_step, gat_output_dim])
    ep_action = np.zeros([ep_max_step])
    ep_reward = np.zeros([ep_max_step])
    ep_next_ope_fea = np.zeros([ep_max_step, ope_num + 2, ope_dim])
    ep_next_ope_adj = np.zeros([ep_max_step, ope_num + 2, ope_num + 2])
    ep_next_insert_adj = np.zeros([ep_max_step, insert_num, ope_num + 2])
    ep_next_insert_mask = np.zeros([ep_max_step, ope_num * insert_num])
    ep_hn = np.zeros([ep_max_step, gat_output_dim])
    ep_cn = np.zeros([ep_max_step, gat_output_dim])
    ep_done = np.zeros([ep_max_step])

    total_step = 0
    ave_reward = 0
    ave_decady = 0
    ave_cmax = 0
    ep_count = 0
    for episode in range(max_episodes):
        ope_fea, ope_adj, insert_adj, insert_mask, init_cmax = env.reset(init_mode='Recurrent', mode=reset_mode)
        reward_info['init_cmax'] = init_cmax
        cmax_buffer = [init_cmax]
        h = np.zeros((1, gat_output_dim))
        c = np.zeros((1, gat_output_dim))
        for step in range(ep_max_step):
            ep_ope_fea[step] = ope_fea
            ep_ope_adj[step] = ope_adj
            ep_insert_adj[step] = insert_adj
            ep_insert_mask[step] = insert_mask
            ep_h[step] = h
            ep_c[step] = c
            action, hn, cn = agent.choose_action(ope_fea, ope_adj, insert_adj, insert_mask, h, c, isTrain=True)
            next_ope_fea, next_ope_adj, next_insert_adj, next_insert_mask, cmax, done, _ = env.step(action)
            total_step += 1
            ep_next_ope_fea[step] = next_ope_fea
            ep_next_ope_adj[step] = next_ope_adj
            ep_next_insert_adj[step] = next_insert_adj
            ep_next_insert_mask[step] = next_insert_mask
            ep_hn[step] = hn
            ep_cn[step] = cn
            ep_done[step] = done
            ep_action[step] = action

            cmax_buffer.append(cmax)

            ope_fea = next_ope_fea
            ope_adj = next_ope_adj
            insert_adj = next_insert_adj
            insert_mask = next_insert_mask
            h = hn
            c = cn

        rewards = env.reward(reward_type, cmax_buffer, reward_info)
        ep_buffer_size = len(rewards)
        ep_reward[:ep_buffer_size] = rewards
        agent.memory.store(ep_ope_fea[:ep_buffer_size],
                           ep_ope_adj[:ep_buffer_size],
                           ep_insert_adj[:ep_buffer_size],
                           ep_insert_mask[:ep_buffer_size],
                           ep_h[:ep_buffer_size],
                           ep_c[:ep_buffer_size],
                           ep_action[:ep_buffer_size],
                           ep_reward[:ep_buffer_size],
                           ep_next_ope_fea[:ep_buffer_size],
                           ep_next_ope_adj[:ep_buffer_size],
                           ep_next_insert_adj[:ep_buffer_size],
                           ep_next_insert_mask[:ep_buffer_size],
                           ep_hn[:ep_buffer_size],
                           ep_cn[:ep_buffer_size],
                           ep_done[:ep_buffer_size],
                           ep_buffer_size,
                           )

        best_cmax = min(cmax_buffer)
        cmax_decady = init_cmax - best_cmax
        reward_sum = round(sum(rewards), 2)
        Summer.add_scalar("Reward", reward_sum, episode)
        Summer.add_scalar("Cmax decady", cmax_decady, episode)
        # print(
        #     "Episode : {} \t\t Episode Reward : {:6.2f} \t\t Cmax decady: {} \t\t Best Cmax : {}".format(
        #         episode,
        #         reward_sum,
        #         cmax_decady, best_cmax))
        ave_reward += reward_sum
        ave_decady += cmax_decady
        ave_cmax += best_cmax
        ep_count += 1
        if ep_count % len(env.graph_list) == 0:
            print(
                "Episode : {} \t\t Avenger Reward : {:6.2f} \t\t Avenger decady: {:6.2f} \t\t Avenger Cmax : {:6.2f}".format(
                    ep_count,
                    ave_reward / len(env.graph_list),
                    ave_decady / len(env.graph_list),
                    ave_cmax / len(env.graph_list),
                ))
            ave_reward = 0
            ave_decady = 0
            ave_cmax = 0
        if agent.memory.total_count > agent.memory.max_size:
            for i in range(5):
                agent.learn()
        if episode % 500 == 0:
            save_model_info['episode'] = episode
            agent.save_models(save_model_info)

if __name__ == '__main__':
    main()
