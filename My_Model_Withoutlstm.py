# import numpy as np
import torch
# import torch.autograd as autograd
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import math


class My_Model_Withoutlstm(nn.Module):  # 一个模块处理一个pair 的所有path

    def __init__(self, node_size, input_dim, out_dim, pre_embedding, all_variables):
        super(My_Model_Withoutlstm, self).__init__()
        self.node_size = node_size
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.pre_embedding = pre_embedding  # 传进来的是随机初始化的pre embedding
        self.embedding = nn.Embedding(node_size, input_dim)  # node size 表示一共多少个node  input dim 是维度
        self.embedding.weight = nn.Parameter(
            pre_embedding)  # 将一个不可训练的参数变成一个可训练的参数  在训练中优化  所以load——pre——embedding  方法是对齐？

        self.linear = nn.Linear(input_dim, out_dim, bias=True)
        self.linear2 = nn.Linear(input_dim, out_dim, bias=True)
        self.relu = nn.ReLU()
        self.all_variables = all_variables
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=2, dim_feedforward=16)
        self.transformerEncoder = nn.TransformerEncoder(num_layers=1, encoder_layer=self.transformerEncoderLayer)
        self.lstm = nn.LSTM(input_dim, input_dim)

    # from path to meta-path
    # def represent_of_meta_path(self,UIUI_in_id,UIAI_in_id,UIDI_in_id,UIGI_in_id):

    def poolings(self, Matrix_of_meta_paths):
        pool_size = Matrix_of_meta_paths.shape[0]
        pool = nn.MaxPool2d((pool_size, 1), stride=(pool_size, 1))
        max_pool = pool(Matrix_of_meta_paths.view(1, pool_size, self.input_dim))
        max_pool.view(1, self.input_dim)
        return max_pool

    # first layer
    def representation_of_paths(self, paths_in_id):  #

        paths_size = len(paths_in_id)

        representation_of_paths = Variable(torch.Tensor([1,self.input_dim]), requires_grad=True)

        for i in range(paths_size):  # 一个user item pair

            path = paths_in_id[i]
            path = path.squeeze()
            path_size = len(path)
            path_embedding = self.embedding(path)  # 知道了一个 path的 embedding
            path_embedding = path_embedding.view(1, path_size, self.input_dim)  # [4,1(batch_size),10]

            # shape 7 1(batch size) 10
            if torch.cuda.is_available():
                path_embedding = path_embedding.cuda()

            representation_of_one_path = self.transformerEncoder(path_embedding).view(path_size,1)  # [4,1,10]->[4,10]
            representation_of_one_path = self.lstm
            if i == 0:
                representation_of_paths = representation_of_one_path
            else:
                representation_of_paths = torch.cat((representation_of_paths, representation_of_one_path),0)

        if torch.cuda.is_available():
            representation_of_paths.cuda()

        return representation_of_paths.view(paths_size,self.input_dim)

    # end

    # second layer
    # def get_representation_of_meta_path(self,meta_path_in_id):

    def MLP(self, out):
        out = self.linear(out)
        out = F.sigmoid(out)

        return out

    def forward(self,paths_in_id):  # 前向传播的一个pair 的 所有 path  当使用model时需要传入 paths_between_one_pair_id
        # 在training 中使用了 paths_between_one_pair_id

        paths_representation = self.representation_of_one_meta_path(paths_in_id).cuda()




        out_for_MLP = self.poolings(paths_representation)

        out_for_MLP = out_for_MLP.cuda()

        out_for_output = self.MLP(out_for_MLP)

        return out_for_output
