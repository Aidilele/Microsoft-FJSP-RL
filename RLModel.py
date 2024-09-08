import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
from Enviroment import *
import RawDataProcess
from Node import *
import time
import sys
from copy import deepcopy
import numpy
from UsefulFunction import check_circle
from Graph import *
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, input_dim, hidden_num, output_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_num),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_num),
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_num),
            nn.Linear(hidden_num, output_dim),
            nn.ELU()
        )
    def forward(self, ope, insert, mask):
        insert_num = insert.shape[-2]
        ope_num = ope.shape[-2]
        fea_dim=insert.shape[-1]
        s = torch.cat([ope.repeat(1, 1, insert_num).view(-1, insert_num * ope_num, fea_dim),
                       insert.repeat(1, ope_num, 1).view(-1, insert_num * ope_num, fea_dim)], dim=-1)
        weight = self.net(s).squeeze(dim=-1)
        act_prob = F.softmax(weight + mask, dim=1)
        return act_prob


class Critic(nn.Module):

    def __init__(self, input_dim, hidden_num, output_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, output_dim),
        )

    def forward(self, ope, insert):
        insert_num = insert.shape[-2]
        ope_num = ope.shape[-2]
        fea_dim=insert.shape[-1]
        s = torch.cat([ope.repeat(1, 1, insert_num).view(-1, insert_num * ope_num, fea_dim),
                       insert.repeat(1, ope_num, 1).view(-1, insert_num * ope_num, fea_dim)], dim=-1)
        value = self.net(s).squeeze(-1)
        value_pool = value.mean(dim=-1)
        return value_pool
