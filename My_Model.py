#import numpy as np
import torch
#import torch.autograd as autograd
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import math

class Mymodel(nn.Module):  # 一个模块处理一个pair 的所有path
    
    def __init__(self, node_size, input_dim, hidden_dim, out_dim, pre_embedding, atten_dim, all_variables, nonlinearity='relu', n_layers=1,
                 dropout=0.5):
        super(Mymodel, self).__init__()
        self.node_size = node_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.pre_embedding = pre_embedding  # 传进来的是随机初始化的pre embedding
        self.embedding = nn.Embedding(node_size, input_dim)  # node size 表示一共多少个node  input dim 是维度
        self.embedding.weight = nn.Parameter(pre_embedding)  # 将一个不可训练的参数变成一个可训练的参数  在训练中优化  所以load——pre——embedding  方法是对齐？
        self.lstm = nn.LSTM(input_dim, hidden_dim,dropout=dropout)  # 设置 lstm 参数
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)

        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout)
        self.atten_dim = atten_dim
        self.all_variables = all_variables
    #from path to meta-path
    #def represent_of_meta_path(self,UIUI_in_id,UIAI_in_id,UIDI_in_id,UIGI_in_id):



    def poolings(self,Matrix_of_meta_paths):
        pool_size = Matrix_of_meta_paths.shape[0]
        pool = nn.MaxPool2d((pool_size,1),stride=(pool_size,1))
        max_pool = pool(Matrix_of_meta_paths.view(1,pool_size,self.hidden_dim))
        max_pool.view(1,self.hidden_dim)
        return max_pool


    #first layer
    def representatio_of_one_meta_path(self,one_kind_of_meta_path_id): #
        sum_hidden = Variable(torch.Tensor(), requires_grad=True)

        paths_size = len(one_kind_of_meta_path_id)
        for i in range(paths_size):  # 一个user item pair
            path = one_kind_of_meta_path_id[i]
            path = path.squeeze()
            path_size = len(path)
            path_embedding = self.embedding(path)
            path_embedding = path_embedding.view(path_size, 1,
                                                 self.input_dim)  # 修改形状  一个 lstm 的iput 是 size of path between one pair
            # shape 7 1(batch size) 10
            if torch.cuda.is_available():
                path_embedding = path_embedding.cuda()
            _, h = self.lstm(path_embedding)
            if i == 0:
                sum_hidden = h[0]  # h[0] 就是一个句子的embedding
                sum_hidden = sum_hidden.view(1,self.hidden_dim)
            else:
                #sum_hidden = torch.cat((sum_hidden, h[0]), 1)  # 在第二维进行拼接  也就是 弄成一个矩阵
                sum_hidden = torch.add(sum_hidden,h[0].view(1,self.hidden_dim))


        sum_hidden = sum_hidden/paths_size

        if torch.cuda.is_available():
            sum_hidden.cuda()

        return sum_hidden
    #end






    # second layer
    #def get_representation_of_meta_path(self,meta_path_in_id):

    def MLP(self,out_for_MLP):
        out = self.linear(out_for_MLP)
        out = torch.sigmoid(out)
        return out







    def forward(self, UIUI_in_id,UIAI_in_id,UIDI_in_id,UIGI_in_id):  # 前向传播的一个pair 的 所有 path  当使用model时需要传入 paths_between_one_pair_id
                                                # 在training 中使用了 paths_between_one_pair_id

        UIUI_representation =self.representatio_of_one_meta_path(UIUI_in_id).cuda()
        UIAI_representation =self.representatio_of_one_meta_path(UIAI_in_id).cuda()
        UIDI_representation =self.representatio_of_one_meta_path(UIDI_in_id).cuda()
        UIGI_representation =self.representatio_of_one_meta_path(UIGI_in_id).cuda()


        Matrix_of_four_meta_path = torch.cat((UIUI_representation, UIAI_representation,UIDI_representation,UIGI_representation), 0)


        out_for_MLP = self.poolings(Matrix_of_four_meta_path)
        #out_for_MLP = self.attention(Matrix_of_four_meta_path,hidden_dim = self.hidden_dim,atten_dim=self.atten_dim)
        out_for_MLP =  out_for_MLP.cuda()

        out_for_output = self.MLP(out_for_MLP)



        return out_for_output
