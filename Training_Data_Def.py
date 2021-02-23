import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader

class MyData(Data.Dataset):


    def __init__(self, training_file):

        self.training_file = training_file





    def __getitem__(self, index):

        user_item_pairs, labels = self.get_training_data(self.training_file)
        user_item_pair = user_item_pairs[index]
        label = labels[index]

        return user_item_pair, label

    def get_training_data(self, training_file):

        self.training_file = training_file

        fr_training = open(training_file, 'r')
        user_item_pairs = []
        labels =[]
        for line in fr_training:
            line_data = line.split('\t')
            user = 'u' + line_data[0]
            item = 'i' + line_data[1]

            label = line_data[2].replace('\n', '')
            user_item_pair = user + ',' + item

            user_item_pairs.append(user_item_pair)
            labels.append(label)

        fr_training.close()

        return user_item_pairs, labels

    def __len__(self):
        user_item_pairs, labels = self.get_training_data(self.training_file)
        length = len(labels)
        return length







