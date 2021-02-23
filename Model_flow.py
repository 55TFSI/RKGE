import torch
from torch.utils.data import Dataset


import numpy as np
import random
#from Model_train import Model_train
#from My_Model import Mymodel
from My_Model_Withoutlstm import My_Model_Withoutlstm
import argparse
from datetime import datetime
from Model_Training import Model_Training
#from ModelEvaluation import ModelEvaluation
from Training_Data_Def import MyData



def load_data(fr_file,istrain): # 传进来的是训练数据
    '''
    load training or test data
    input: user - item rating data
    :return: user -spicific rating data with time stamp
    '''

    data_dict ={}  # 创建一个 空字典
    if istrain == True:
        for line in fr_file:
            lines = line.split('\t')   #每一行去掉换行符   用空格分成 3个 list
            user = 'u' + lines[0]
            item = 'i' + lines[1]
            label = lines[2].replace('\n','')
            if label == '1':
                if user not in data_dict:
                    data_dict.update({user: [item]})
                elif item not in data_dict[user]:
                    data_dict[user].append(item)
            else:
                pass
    else:
        for line in fr_file:
            lines = line.split('\t')   #每一行去掉换行符   用空格分成 3个 list
            user = 'u' + lines[0]
            item = 'i' + lines[1].replace('\n', '')

            if user not in data_dict:
                data_dict.update({user: [item]})
            elif item not in data_dict[user]:
                data_dict[user].append(item)             # 第一次将train 的user-item pair {user : item}


    return data_dict

def load_paths(fr_file):  # 传进来的是 paths    # 返回所有node all_all_variables paths_between_pairs {}
    '''
    load postive or negative paths, map all nodes in paths into ids

    Inputs:
    	@fr_file: positive or negative paths
    	@isPositive: identify fr_file is positive or negative
    '''

    global node_count, all_variables, paths_between_pairs, all_user, all_movie

    for line in fr_file:
        line = line.replace('\n', '')
        lines = line.split(',')
        user = lines[0]
        movie = lines[-1]

        # relation = ['r1','r2','r3','r4','r5','r6','r7','r8']

        if user not in all_user:
            all_user.append(user) # all user all movie 两个list 储存 user 和movie
        if movie not in all_movie:
            all_movie.append(movie)

        key = (user, movie)
        value = []
        path = []

        #if isPositive:
           # if key not in positive_label:
               # positive_label.append(key)             # positive_label 存储的user item 对 （postive）

        for node in lines:
            if node not in all_variables:
                all_variables.update({node: node_count})
                node_count = node_count + 1
            path.append(node)                            # all  variables  = {node : node_count}

        if key not in paths_between_pairs:
            value.append(path)
            paths_between_pairs.update({key: value})   #paths_between_pairs { (user-item pairs): (某一个pair 所有的路径)}
        else:
            paths_between_pairs[key].append(path)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=''' MY BS model ''')

    '''

    	Current parameter Settings: 
    	for MovieLens in terms of [input_dim, hidden_dim, out_dim, iteration, learning_rate, optimizer] is [10, 16, 1, 5, 0.2/0.1, SGD]

    	You can change optimizer in the LSTMTrain class
    	'''

    # 模型参数们
    parser.add_argument('--trainingmetapathpostive', type=str, dest='positive_paths', default='data/positive_paths.txt')


    parser.add_argument('--trainingNmetapathnegative', type=str, dest='negative_paths', default='data/negative_paths.txt')



    parser.add_argument('--inputdim', type=int, dest='input_dim', default=10)
    parser.add_argument('--hiddendim', type=int, dest='hidden_dim', default=10)
    parser.add_argument('--outdim', type=int, dest='out_dim', default=1)
    parser.add_argument('--iteration', type=int, dest='iteration', default=20)
    parser.add_argument('--learingrate', type=float, dest='learning_rate', default=0.2)
    #parser.add_argument('--train', type=str, dest='train_file', default="data/training_all.txt")
    parser.add_argument('--train', type=str, dest='train_file', default="data/traning_for_parameter.txt")

    parser.add_argument('--test', type=str, dest='test_file', default='data/test.txt')
    parser.add_argument('--results', type=str, dest='results', default='data/results.txt')
    parser.add_argument('--attendim', type=int, dest='atten_dim', default=10)
    parser.add_argument('--batchsize', type=int, dest='batch_size', default=256)

    parsed_args = parser.parse_args()


    atten_dim = parsed_args.atten_dim
    input_dim = parsed_args.input_dim
    hidden_dim = parsed_args.hidden_dim
    out_dim = parsed_args.out_dim
    iteration = parsed_args.iteration
    learning_rate = parsed_args.learning_rate

    positive_paths = parsed_args.positive_paths

    negative_paths = parsed_args.negative_paths


    batch_size = parsed_args.batch_size

    train_file = parsed_args.train_file
    test_file = parsed_args.test_file
    results_file = parsed_args.results


    fr_positive_paths = open(positive_paths, 'r')
    fr_negative_paths = open(negative_paths,'r')


    fr_train = open(train_file, 'r')
    fr_test = open(test_file, 'r')
    fw_results = open(results_file, 'w')
    #end


    # 初始化
    node_count = 0  # count the number of all entities (user, movie and attributes)
    all_variables = {}  # save variable and corresponding id
    paths_between_pairs = {}


    all_user = []  # save all the users
    all_movie = []  # save all the movies
    #end



    #load datas
    start_time = datetime.now()
    load_paths(fr_positive_paths)   # 返回一个字典 是所有的 user-item pair
    load_paths(fr_negative_paths)
    print('The number of all variables is :' + str(len(all_variables)))
    end_time = datetime.now()
    duration = end_time - start_time
    print('the duration for loading user path is ' + str(duration) + '\n')
    #end



    #pre-embedding or embedding
    node_size = len(all_variables)
    np.random.seed(5)
    pre_embedding = np.random.rand(node_size, input_dim) # embeddings for all nodes

    pre_embedding = torch.FloatTensor(pre_embedding)  # 把predmbedding 变成了 tensor
    #pre_embedding = torch.nn.Embedding(node_size,input_dim)
    # end

    # 初始化模型  传进去的 pre-embedding
    model = My_Model_Withoutlstm(node_size, input_dim, out_dim, pre_embedding,all_variables,)  # prembedding 是随机初始化的向量
    # model = LSTMTagger(node_size, input_dim, hidden_dim, out_dim, pre_embedding)
    #model = My_Model_Withoutlstm(node_size, input_dim, hidden_dim, out_dim, pre_embedding, atten_dim,
                    #all_variables)  # prembedding 是随机初始化的向量
    if torch.cuda.is_available():
        model = model.cuda()

    #end


    # training and evaluatie every epoch
    start_time = datetime.now()
    embedding_dict = {}
    test_dict = load_data(fr_test,False)
    train_dict = load_data(fr_train,True)
    count = 0
    model_train = Model_Training(model, iteration, learning_rate, train_file,test_file,paths_between_pairs,
                              all_variables, all_user, all_movie, batch_size, embedding_dict,test_dict,train_dict,count)

    embedding_dict = model_train.train()
    end_time = datetime.now()
    duration = end_time-start_time

    print('the duration for model training is ' + str(duration) + '\n')
    fr_negative_paths.close()
    fr_positive_paths.close()




    fr_train.close()
    fr_test.close()
    fw_results.close()
