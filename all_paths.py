
import argparse
from  random import sample

def load_data(fr_file,fw_file):
    all_users = []
    all_moives = []

    for lines in fr_file:
        if lines.startswith('i'):
            all_moives.append(lines.replace('\n',''))
        if lines.startswith('u'):
            all_users.append(lines.replace('\n',''))

    for users in all_users:
        item_candidate = sample(all_moives,300)
        for items in item_candidate:
            line = users +','+  items + '\n'

            fw_file.write(line)






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=''' mine all paths''')


    parser.add_argument('--all_nodes', type=str, dest='all_nodes', default='data/all_nodes.txt')
    parser.add_argument('--candidate_user_items', type=str, dest='candidate_user_items', default='data/candidate_user_items.txt')

    parsed_args = parser.parse_args()

    all_nodes = parsed_args.all_nodes
    candidate_user_items = parsed_args.candidate_user_items


    fr_file = open(all_nodes, 'r')
    fw_file = open(candidate_user_items,'w')
    load_data(fr_file,fw_file)


    fr_file.close()
    fw_file.close()
