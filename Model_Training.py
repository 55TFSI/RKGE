import numpy as np
import torch
import torch.autograd as autograd
from torch import nn, optim
from torch.autograd import Variable
# test
import torch
import torch.utils.data as Data
import datetime
import heapq
from Training_Data_Def import MyData
from torch.utils.data import DataLoader
from Model_Evaluation import ModelEvaluation

class Model_Training(object):
    '''
    lstm training process
    '''

    def __init__(self, model, iteration, learning_rate, training_data,testing_data,paths_between_pairs,
                 all_variables, all_user, all_movie,batch_size,embedding_dict,test_dict,train_dict,count):
        self.model = model
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.training_data = training_data
        self.paths_between_pairs = paths_between_pairs
        self.all_variables = all_variables
        self.all_user = all_user
        self.all_movie = all_movie
        self.testing_data =testing_data
        self.batch_size = batch_size
        self.embedding_dict = embedding_dict
        self.test_dict = test_dict
        self.train_dict = train_dict
        self.count = count



    def load_paths_between_a_pair(self,tr_data): # pair 是一个list [user,item]
        data = tr_data.split(',') # 用空格分开
        user = data[0]
        item = data[1].replace('\n', '')
        if (user,item) in self.paths_between_pairs:
            paths_between_a_pair = self.paths_between_pairs[(user,item)]
        else:
            paths_between_a_pair = self.paths_between_pairs[('u196','i242')]
            print((user,item))
            self.count = self.count + 1

        return paths_between_a_pair


    def  assign_meta_path_between_a_pair(self,paths):
        UIUI = []
        UIAI = []
        UIDI = []
        UIGI = []
        for path in paths:
            sign = path[2]
            if sign.startswith('u'):
                UIUI.append(path)
            elif sign.startswith('a'):
                UIAI.append(path)
            elif sign.startswith('d'):
                UIDI.append(path)
            else:
                UIGI.append(path)
        return UIUI,UIAI,UIDI,UIGI



    def load_paths_between_one_pair_id(self,path):  # map a path to id
        one_path_id = []
        nodes_id_in_a_path = [self.all_variables[x] for x in path]
        one_path_id.append(nodes_id_in_a_path)

        return  one_path_id


    # update embedding dict
    def dump_post_embedding(self):

        node_list = self.all_user + self.all_movie

        for node in node_list:
            #node_id = torch.LongTensor()  # all_variables 呆一会儿写
            node_id = torch.LongTensor([int(self.all_variables[node])])
            node_id = Variable(node_id)

            if torch.cuda.is_available():
                node_id = node_id.cuda()

            #node_embedding = self.model.embedding(node_id).squeeze().cpu().data.numpy()
            node_embedding = self.model.embedding(node_id).squeeze().cpu().data.numpy()

            self.embedding_dict.update({node: node_embedding})

        return self.embedding_dict


    def evaluate_in_test_data(self):
        embedding_dict = self.embedding_dict
        model_evaluate = ModelEvaluation(embedding_dict,self.all_movie,self.train_dict,self.test_dict)
        top_score_dict = model_evaluate.calculate_ranking_score()

        precision_1, _ = model_evaluate.calculate_results(top_score_dict, 1)
        precision_5, _ = model_evaluate.calculate_results(top_score_dict, 5)
        precision_10, mrr_10 = model_evaluate.calculate_results(top_score_dict, 10)

        print('precision_1 = ',precision_1,'precision_5 = ',precision_5,'precision_10 = ',precision_10,
              'mrr_10 = ', mrr_10)










    def train(self):
        # loss funtion and optimizer
        criterion = nn.BCELoss()  # BCE loss
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        #end


        # load data and data loader
        train_data = MyData(self.training_data)
        train_data_loader = DataLoader(train_data, self.batch_size, shuffle=True)

        #end

        #initialize a Evaluate class


        #train
        for epoch in range (self.iteration):


            for steps ,(tr_data, tr_label) in enumerate(train_data_loader):
                running_loss = 0.0
                loss = 0.0
                for i in range(len(tr_data)):
                    paths_in_id = []
                    paths = self.load_paths_between_a_pair(tr_data[i])  # tr_data  : paths between 1 pair
                   #UIUI,UIAI,UIDI,UIGI = self.assign_meta_path_between_a_pair(paths)
                    for path in paths:
                        path_in_id = self.load_paths_between_one_pair_id(path)
                        paths_in_id.append(path_in_id)
                    paths_in_id = np.array(paths_in_id)
                    paths_in_id = Variable(torch.LongTensor(paths_in_id))
                    paths_in_id = paths_in_id.cuda()

                    out = self.model(paths_in_id)
                    #out = self.model(paths_in_id)

                    loss = criterion(out.cpu(), torch.Tensor([int(tr_label[i])]))
                    running_loss += loss.item()  # * int(tr_label[i])

                print('epoch = ', epoch ,'batch = ',steps, 'runing loss = ', running_loss, 'data loss = ', self.count)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #update embedding
            self.embedding_dict = self.dump_post_embedding()
            #end


            #val the model per epoch
            self.evaluate_in_test_data()
            #print(self.embedding_dict['r1'])
            #end
        return  self.dump_post_embedding()







