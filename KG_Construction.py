# this is to build a knowledge graph that entites including users directors actors genres
# then min path between path and items

import argparse
import networkx as nx
import random




def load_data(file):
    '''
        load training data

        Input:
            @file: training (positive) data or negative data

        Output:
            @data: pairs containing positive or negative interaction data
        '''
    data = []

    for line in file:
        lines = line.split('\t')
        user = lines[0]
        movie = lines[1]
        label = lines[2].replace('\n', '')
        data.append((user, movie, label))

    return data



def add_user_item_interaction_into_graph(ratings_in_training_data):
    '''
    this method is to add user_movie interaction into graph  including relation
    input:
        @postivae_rating : user_interaction data
    :return:
        the graph built graph (with user_interaction info)

    '''

    interaction_dic = {'interact' : '1' ,'_interacted': '2'}

    graph = nx.DiGraph()
    #user_list= []
    #item_list =[]
    #number_of_user = number_of_item = 0


    for pairs in ratings_in_training_data:
        user_id = pairs[0]
        item_id = pairs[1]
        label = pairs[2]

        if label == '1':
            user_node = 'u' + user_id
            item_node = 'i' + item_id
            graph.add_node(user_node)
            graph.add_node(item_node)
            graph.add_edge(user_node,item_node, relation = 'r' + interaction_dic['interact'])
            graph.add_edge(item_node, user_node, relation = 'r' + interaction_dic['_interacted'])

        #if not user_id in user_list:
            #user_list.append(user_id)
            #number_of_user= number_of_user + 1

        #if not item_id in item_list:
            #item_list.append(item_id)
            #number_of_item = number_of_item + 1

    #print('number of item is :',number_of_item)
    #print('number of user is :', number_of_user)



    return graph

def add_auxiliary_data_into_graph(fr_auxiliary, Graph):
    '''
    this method is to add auxiliary data(genre actor director) in to graph
    input: auxiliary mapping
           graph with user-item interaction

    output:knowledge
    '''

    relation_dic = {'direct' : '3', '_direct_by ': '4', 'act': '5', '_act_by':'6',
                    'include_of':'7', 'belongs_to': '8'}
    for line in fr_auxiliary:
        lines = line.replace('\n','').split('|')
        if len(lines) != 4: continue

        movie_id = lines[0]
        genre_list =lines[1].split(',')
        director_list = lines[2].split(',')
        actor_list = lines[3].split(',')

        # add moves nodes that not occured in user-item interaction data\

        movie_node = 'i' + movie_id
        if not Graph.has_node(movie_node):
            Graph.add_node(movie_node)


        #add genres in to knowledge graph
        #as genre connection is too dense, we add one genre to avoid over-emphasizing its effect
        genre_id = genre_list[0]
        genre_node = 'g' + genre_id
        if not Graph.has_node(genre_node):
            Graph.add_node(genre_node)
        Graph.add_edge(movie_node,genre_node, relation = 'r' + relation_dic['belongs_to'])
        Graph.add_edge(genre_node, movie_node, relation='r' + relation_dic['include_of'])



        #add dirctors in knowledge graph
        for director_id in director_list:
            director_node = 'd' + director_id
            if not Graph.has_node(director_node):
                Graph.add_node(director_node)
            Graph.add_edge(movie_node,director_node, relation = 'r' + relation_dic['direct'])
            Graph.add_edge(director_node, movie_node, relation='r' + relation_dic['_direct_by '])


        #add actors into knowledge graph
        for actor_id in actor_list:
            actor_node = 'a'+ actor_id
            if not Graph.has_node(actor_node):
                Graph.add_node(actor_node)
            Graph.add_edge(actor_node,movie_node,relation = 'r' + relation_dic['act'])
            Graph.add_edge(movie_node,actor_node,relation = 'r' + relation_dic['_act_by'])

    return  Graph

def print_graph_statistic(Graph,fw_all_nodes):
    '''
    output the statistic info of the graph

    Input:
        @Graph: the built graph
    Output: all the nodes and (edges) in knowledge G
    '''

    for node in Graph.nodes():
        line = node + '\n'
        fw_all_nodes.write(line)

    print('The knowledge graph has been built completely \n')
    print('The number of nodes is:  ' + str(len(Graph.nodes()))+ ' \n')
    print('The number of edges is  ' + str(len(Graph.edges()))+ ' \n')



def mine_paths_between_nodes(Graph, user_node, movie_node, maxLen,fw_file):
    '''
    mine qualified paths between user and movie nodes, and get sampled paths between nodes

    Inputs:
        @user_node: user node
        @movie_node: movie node
        @maxLen: path length
        @fw_file: the output file for the mined paths
    '''
    UIUI = []
    UIAI =[]
    UIDI = []
    UIGI = []
    all_path = []

    for path in nx.all_simple_paths(Graph, source= user_node, target= movie_node, cutoff= maxLen):

        if len(path) == maxLen + 1:
            if path[2][0] == 'u':
                UIUI.append(path)

            if path[2][0] == 'a':
                UIAI.append(path)

            if path[2][0] == 'd':
                UIDI.append(path)
            if path[2][0] == 'g':
                UIGI.append(path)

    path_size = 6
    UIUI_size = len(UIUI)
    UIAI_size = len(UIAI)
    UIDI_size = len(UIDI)
    UIGI_size = len(UIGI)

    # as there is a huge number of p conneathscted user-movie nodes, we get randomly sampled paths
    # random sample can better balance the data distribution and model complexity
    if UIUI_size > path_size:
        random.shuffle(UIUI)
        UIUI = UIUI[:path_size]

    if UIAI_size >= path_size:
        random.shuffle(UIAI)
        UIAI = UIAI[:path_size]




    if UIDI_size >= path_size:
        random.shuffle(UIDI)
        UIDI = UIDI[:path_size]



    if UIGI_size > path_size:
        random.shuffle(UIGI)
        UIGI = UIGI[:path_size]

    all_path.extend(UIUI)
    all_path.extend(UIAI)
    all_path.extend(UIDI)
    all_path.extend(UIGI)



    for path in all_path:
        line = "," .join(path) + '\n'

        fw_file.write(line)



def dump_paths(Graph, ratings_in_training_data, maxLen, fw_file):
    '''
    dump the postive or negative paths

    Inputs:
        @Graph: the well-built knowledge graph
        @rating_pair: positive_rating or negative_rating
        @maxLen: path length
        @sample_size: size of sampled paths between user-movie nodes
    '''

    for pair in ratings_in_training_data:
        user_id = pair[0]
        movie_id = pair[1]

        user_node = 'u' + user_id
        movie_node = 'i' + movie_id

        if Graph.has_node(user_node) and Graph.has_node(movie_node):
            mine_paths_between_nodes(Graph, user_node, movie_node, maxLen, fw_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=''' Build Knowledge Graph and Mine the Connected Paths''')


    parser.add_argument('--training', type=str, dest='training_file', default='data/training-positive.txt')
    parser.add_argument('--negative', type=str, dest='negative_file', default='data/negative.txt')
    parser.add_argument('--auxiliary', type=str, dest='auxiliary_file', default='data/auxiliary-mapping.txt')


    parser.add_argument('--pathlength', type=int, dest='path_length', default=3,
                        help='length of paths with choices [3,5,7,9,11]')

    parser.add_argument('--negetive_paths', type=str, dest='negative_paths', default='data/negative_paths.txt')
    parser.add_argument('--positive_paths', type=str, dest='positive_paths', default='data/positive_paths.txt')
    parser.add_argument('--all_nodes', type=str, dest='all_nodes', default='data/all_nodes.txt')


    parsed_args = parser.parse_args()
    training_file = parsed_args.training_file
    negative_file =parsed_args.negative_file
    auxiliary_file = parsed_args.auxiliary_file
    path_length = parsed_args.path_length

    positive_paths = parsed_args.positive_paths
    negative_paths = parsed_args.negative_paths
    all_nodes = parsed_args.all_nodes



    fr_training = open(training_file, 'r')
    fr_negative = open(negative_file, 'r')
    fr_auxiliary = open(auxiliary_file, 'r')

    fw_file1 = open(positive_paths, 'w')
    fw_file2 = open(negative_paths, 'w')
    fw_all_nodes = open(all_nodes, 'w')



    # fw_negative_path = open(negative_path, 'w')
    #fw_all_nodes = open(all_nodes,'w')

    training_positive = load_data(fr_training)
    training_negative = load_data(fr_negative)





    # print('The number of user-movie interaction data is:  ' + str(len(positive_rating)) + ' \n')
    Graph = add_user_item_interaction_into_graph(training_positive)
    Graph = add_auxiliary_data_into_graph(fr_auxiliary, Graph)



    print_graph_statistic(Graph,fw_all_nodes)

    dump_paths(Graph, training_positive, path_length,fw_file1)
    dump_paths(Graph,training_negative,path_length,fw_file2)



    fw_file1.close()
    fw_file2.close()

    fr_training.close()
    fr_auxiliary.close()

    # fw_negative_path.close()
