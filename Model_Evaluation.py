import heapq
import numpy as np
import torch
from torch.autograd import Variable

class ModelEvaluation(object):
    '''
    recurrent neural network evaluation
    '''

    #def __init__(self, embedding_dict, all_movie, train_dict, test_dict):
    def __init__(self, embedding_dict, all_movie, train_dict, test_dict):
        super(ModelEvaluation, self).__init__()
        self.embedding_dict = embedding_dict
        self.all_movie = all_movie
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.mrr = 0.0
        #self.model = model
        #self.paths_between_pairs = paths_between_pairs
    ###

    def calculate_ranking_score(self):
        '''
        calculate ranking score of unrated movies for each user
        '''
        score_dict = {}
        top_score_dict = {}

        for user in self.test_dict:  #对于 在 test_dict 的所有用户
            if user in self.embedding_dict and user in self.train_dict:  # 对于一个有 embedding 的user （训练过的）
                for movie in self.all_movie: # 对 一个 user 找所有的item
                    if (movie not in self.train_dict[user]) and movie in self.embedding_dict:   # 对于  在训练集中 用户没交互过的user
                        embedding_user = self.embedding_dict[user]
                        embedding_movie = self.embedding_dict[movie]



                        score = np.dot(embedding_user, embedding_movie)
                        #paths_between_one_pair_id =
                        #score =
                        if user not in score_dict:
                            score_dict.update({user: {movie: score}})
                        else:
                            score_dict[user].update({movie: score})



                if user in score_dict and len(score_dict[user]) > 1:
                    item_score_list = score_dict[user]
                    k = min(len(item_score_list), 15)  # to speed up the ranking process, we only find the top 15 movies
                    top_item_list = heapq.nlargest(k, item_score_list, key=item_score_list.get) # checked
                    top_score_dict.update({user: top_item_list})

        print('len of  embedding dic is :', len(self.embedding_dict))

        return top_score_dict




    def calculate_results(self, top_score_dict, k):
        '''
        calculate the final results: pre@k and mrr
        '''
        precision = 0.0
        isMrr = False
        if k == 10: isMrr = True

        user_size = 0
        for user in self.test_dict:
            if user in top_score_dict:
                user_size = user_size + 1
                candidate_item = top_score_dict[user]
                candidate_size = len(candidate_item)
                hit = 0
                min_len = min(candidate_size, k)
                for i in range(min_len):
                    if candidate_item[i] in self.test_dict[user]:
                        hit = hit + 1
                        if isMrr: self.mrr += float(1 / (i + 1))
                hit_ratio = float(hit / min_len)
                precision += hit_ratio

        precision = precision / user_size
        print('precision@' + str(k) + ' is: ' + str(precision))

        if isMrr:
            self.mrr = self.mrr / user_size
            print('mrr@' + str(k) + ' is: ' + str(self.mrr))

        return precision, self.mrr