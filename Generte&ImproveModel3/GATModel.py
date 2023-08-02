import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
from RLModel import Actor
from torch.distributions import Categorical


class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        batch = inp.shape[0]
        h = torch.matmul(inp, self.W)  # [N, out_features]
        # N = h.size()[0]  # N 图的节点数
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=-1).view(N, -1, 2 * self.out_features)

        N = inp.shape[1]
        M = adj.shape[1]  # N 图的节点数
        a_input = torch.cat([h.repeat(1, 1, M).view(batch, M * N, -1), h.repeat(1, M, 1)], dim=-1).view(batch, M, -1,
                                                                                                        2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=-1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.leaky_relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiGAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_output, dropout=0.0, alpha=0.2, n_heads=4):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(MultiGAT, self).__init__()
        self.dropout = dropout
        # 定义multi-head的图注意力层
        self.attentions = [GATLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = nn.Linear(n_heads * n_hid, n_output)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.tanh(self.out_att(x))  # 输出并激活
        return x  # log_softmax速度变快，保持数值稳定


class OpeModel(nn.Module):

    def __init__(self, input_dim, n_hidden, n_output, layer_num=4):
        super(OpeModel, self).__init__()
        self.ope_fea = nn.ModuleList()

        first_layer = MultiGAT(input_dim, n_hidden, n_output)
        self.ope_fea.append(first_layer)
        for _ in range(layer_num):
            layer = MultiGAT(n_output, n_hidden, n_output)
            self.ope_fea.append(layer)

    def forward(self, fea, adj):
        for module in self.ope_fea:
            fea = module(fea, adj)
        pool = fea.mean(dim=-2)
        # std=fea.std(dim=-2).repeat(1,fea.shape[-2]).view(-1,fea.shape[-2],fea.shape[-1])
        # fea=(fea-pool)/std
        output = torch.cat([fea, pool.repeat(1, fea.shape[1]).view(fea.shape[0], fea.shape[1], -1)], -1)
        return fea


class InsertPositionModel(nn.Module):

    def __init__(self, input_dim, n_hidden, n_output, layer_num=4):
        super(InsertPositionModel, self).__init__()
        self.insert_position_fea = nn.ModuleList()

        first_layer = MultiGAT(input_dim, n_hidden, n_output)
        self.insert_position_fea.append(first_layer)

    def forward(self, fea, adj):
        for module in self.insert_position_fea:
            fea = module(fea, adj)
        pool = fea.mean(dim=-2)
        # std = fea.std(dim=-2).repeat(1, fea.shape[-2]).view(-1, fea.shape[-2], fea.shape[-1])
        # fea = (fea - pool) / std
        output = torch.cat([fea, pool.repeat(1, fea.shape[1]).view(fea.shape[0], fea.shape[1], -1)], -1)
        return fea


class MachineModel(nn.Module):

    def __init__(self, input_dim, n_hidden, n_output, layer_num=4):
        super(MachineModel, self).__init__()
        self.machine_fea = nn.ModuleList()
        first_layer = MultiGAT(input_dim, n_hidden, n_output)
        self.machine_fea.append(first_layer)
        for _ in range(layer_num):
            layer = MultiGAT(n_output, n_hidden, n_output)
            self.machine_fea.append(layer)

    def forward(self, fea, adj):
        for module in self.ope_fea:
            fea = module(fea, adj)
        pool = fea.mean(dim=-2)
        return fea, pool


if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)
    env = FJSPEnviroment()
    ope_fea, ope_adj, insert_adj, insert_mask, init_cmax = env.reset()
    ope_fea = torch.tensor(ope_fea,dtype=torch.float32)
    ope_adj = torch.tensor(ope_adj,dtype=torch.int)
    insert_adj = torch.tensor(insert_adj,dtype=torch.int)
    insert_mask = torch.tensor(insert_mask)
    # ope_fea = np.stack([ope_fea[0], ope_fea[0]])
    # ope_adj = np.stack([ope_adj[0], ope_adj[0]])
    # insert_adj = np.stack([insert_adj[0], insert_adj[0]])
    # insert_mask = np.stack([insert_mask[0], insert_mask[0]])
    openet = OpeModel(7, 8, 8, 4)
    insnet = InsertPositionModel(16, 8, 8, 4)
    policy = Actor(32, 32, 1)
    for i in range(500):
        ope_state = openet(ope_fea, ope_adj)
        ope_state_np = ope_state.detach().numpy()
        insert_state = insnet(ope_state, insert_adj)
        insert_state_np = insert_state.detach().numpy()
        a_prob = policy(ope_state, insert_state, insert_mask)
        dist = Categorical(a_prob)
        action = dist.sample().detach().numpy()
        ope_fea, ope_adj, insert_adj, insert_mask, cmax, _, _ = env.step(action[0])
        print(i, ':', cmax)
