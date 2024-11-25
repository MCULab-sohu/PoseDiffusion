from torch import nn
import torch
import torch.nn.init as init
import math

class SpatialBlock(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatialBlock, self).__init__()
        # 定义可学习的参数 theta b
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # 用kaiming_uniform_初始化 theta 参数
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        # 计算fan_in 值
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        # 计算初始化边界
        bound = 1 / math.sqrt(fan_in)
        # 用均匀分布初始化偏置 b
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        # x:输入张量，形状为 [batch_size, c_in, time, n_nodes],用于本任务就是[bs, heatmap_len, heatmap_len, channels]
        # Lk:拉普拉斯矩阵或其幂，形状为 [3, n_nodes, n_nodes]或[n_nodes, n_nodes]
        if len(Lk.shape) == 2:  # if supports_len == 1:
            Lk = Lk.unsqueeze(0)
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # [b, c_out, time, n_nodes]
        return torch.relu(x_gc + x)