import numpy as np
import random
from copy import deepcopy


#
# class RolloutBuffer:
#     def __init__(self, buffer_size=1024):
#
#         self.buffer = [None] * buffer_size
#         self.buffer_count = 0
#         self.buffer_size = buffer_size
#         self.sample_flag = False
#
#     def step_store(self, ope_fea, ope_adj, action, reward, ope_fea_, ope_adj_, done):
#         self.buffer[self.buffer_count] = [ope_fea, ope_adj, action, reward, ope_fea_, ope_adj_, done]
#         self.buffer_count = (self.buffer_count + 1) % self.buffer_size
#         if self.buffer_count == self.buffer_size - 1:
#             self.sample_flag = True
#
#     def episode_store(self, ep):
#         for ope_fea, ope_adj, action, reward, ope_fea_, ope_adj_, done in ep:
#             self.step_store(ope_fea, ope_adj, action, reward, ope_fea_, ope_adj_, done)
#
#     def sample(self, batch=128):
#         actions = [None] * batch
#         ope_fea = [None] * batch
#         ope_adj = [None] * batch
#         rewards = [None] * batch
#         ope_fea_ = [None] * batch
#         ope_adj_ = [None] * batch
#         is_terminals = [None] * batch
#         if self.sample_flag:
#             sample = random.sample(self.buffer, batch)
#             for i in range(len(sample)):
#                 ope_fea[i] = sample[i][0]
#                 ope_adj[i] = sample[i][1]
#                 actions[i] = sample[i][2]
#                 rewards[i] = sample[i][3]
#                 ope_fea_[i] = sample[i][4]
#                 ope_adj_[i] = sample[i][5]
#                 is_terminals[i] = sample[i][6]
#         return ope_fea, ope_adj, actions, rewards, ope_fea_, ope_adj_, is_terminals

class ReplayBuffer:
    def __init__(self, buffer_size=2048, batch_size=256, ope_num=0, machine_num=0, gat_output_dim=0):

        ope_num = ope_num
        ope_dim = 6
        insert_num = ope_num + machine_num

        self.ope_fea = np.zeros([buffer_size, ope_num+2, ope_dim])
        self.ope_adj = np.zeros([buffer_size, ope_num+2, ope_num+2])
        self.insert_adj = np.zeros([buffer_size, insert_num, ope_num+2])
        self.action_mask = np.zeros([buffer_size, ope_num * insert_num])
        self.h = np.zeros([buffer_size, gat_output_dim])
        self.c = np.zeros([buffer_size, gat_output_dim])
        self.action = np.zeros([buffer_size])
        self.reward = np.zeros([buffer_size])
        self.next_ope_fea = np.zeros([buffer_size, ope_num+2, ope_dim])
        self.next_ope_adj = np.zeros([buffer_size, ope_num+2, ope_num+2])
        self.next_insert_adj = np.zeros([buffer_size, insert_num, ope_num+2])
        self.next_action_mask = np.zeros([buffer_size, ope_num * insert_num])
        self.hn = np.zeros([buffer_size, gat_output_dim])
        self.cn = np.zeros([buffer_size, gat_output_dim])
        self.done = np.zeros([buffer_size])
        self.count = 0
        self.total_count = 0
        self.max_size = buffer_size
        self.batch_size = batch_size

    def store(self, ope_fea, ope_adj, insert_adj, action_mask, h, c,
              action,
              reward,
              next_ope_fea, next_ope_adj, next_insert_adj, next_action_mask, hn, cn,
              done,
              ep_size):
        start = self.count
        end = (self.count + ep_size) % self.max_size
        self.total_count += ep_size
        self.count = end
        if end > start:
            self.ope_fea[start:end] = ope_fea
            self.ope_adj[start:end] = ope_adj
            self.insert_adj[start:end] = insert_adj
            self.action_mask[start:end] = action_mask
            self.h[start:end] = h
            self.c[start:end] = c
            self.action[start:end] = action
            self.reward[start:end] = reward
            self.next_ope_fea[start:end] = next_ope_fea
            self.next_ope_adj[start:end] = next_ope_adj
            self.next_insert_adj[start:end] = next_insert_adj
            self.next_action_mask[start:end] = next_action_mask
            self.hn[start:end] = hn
            self.cn[start:end] = cn
            self.done[start:end] = done
        elif end < start:
            seg_ment = self.max_size - start
            self.ope_fea[start:] = ope_fea[:seg_ment]
            self.ope_adj[start:] = ope_adj[:seg_ment]
            self.insert_adj[start:] = insert_adj[:seg_ment]
            self.action_mask[start:] = action_mask[:seg_ment]
            self.h[start:] = h[:seg_ment]
            self.c[start:] = c[:seg_ment]
            self.action[start:] = action[:seg_ment]
            self.reward[start:] = reward[:seg_ment]
            self.next_ope_fea[start:] = next_ope_fea[:seg_ment]
            self.next_ope_adj[start:] = next_ope_adj[:seg_ment]
            self.next_insert_adj[start:] = next_insert_adj[:seg_ment]
            self.next_action_mask[start:] = next_action_mask[:seg_ment]
            self.hn[start:] = hn[:seg_ment]
            self.cn[start:] = cn[:seg_ment]
            self.done[start:] = done[:seg_ment]

            self.ope_fea[:end] = ope_fea[seg_ment:]
            self.ope_adj[:end] = ope_adj[seg_ment:]
            self.insert_adj[:end] = insert_adj[seg_ment:]
            self.action_mask[:end] = action_mask[seg_ment:]
            self.h[:end] = c[seg_ment:]
            self.c[:end] = c[seg_ment:]
            self.action[:end] = action[seg_ment:]
            self.reward[:end] = reward[seg_ment:]
            self.next_ope_fea[:end] = next_ope_fea[seg_ment:]
            self.next_ope_adj[:end] = next_ope_adj[seg_ment:]
            self.next_insert_adj[:end] = next_insert_adj[seg_ment:]
            self.next_action_mask[:end] = next_action_mask[seg_ment:]
            self.hn[:end] = cn[seg_ment:]
            self.cn[:end] = cn[seg_ment:]
            self.done[:end] = done[seg_ment:]

    def sample(self):
        batch = np.random.choice(self.max_size, self.batch_size, replace=False)
        ope_fea = self.ope_fea[batch]
        ope_adj = self.ope_adj[batch]
        insert_adj = self.insert_adj[batch]
        action_mask = self.action_mask[batch]
        h = self.h[batch]
        c = self.c[batch]
        action = self.action[batch]
        reward = self.reward[batch]
        next_ope_fea = self.next_ope_fea[batch]
        next_ope_adj = self.next_ope_adj[batch]
        next_insert_adj = self.next_insert_adj[batch]
        next_action_mask = self.next_action_mask[batch]
        hn = self.hn[batch]
        cn = self.cn[batch]
        done = self.done[batch]

        return (ope_fea, ope_adj, insert_adj, action_mask, h, c,
                action,
                reward,
                next_ope_fea, next_ope_adj, next_insert_adj, next_action_mask, hn, cn,
                done)
