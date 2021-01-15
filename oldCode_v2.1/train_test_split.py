# -*- coding: utf-8 -*-
from math import ceil
from math import floor
import numpy as np
import os




def main(k, in_path, out_path, random_state = 180):
    read_path = './' + in_path + '/' + str(k)
    write_path = out_path + str(k)
    splits_to_min(read_path, write_path + '/min/', random_state)
    splits_to_more(read_path, write_path + '/more/', random_state)
    
def splits_to_min(read_path, write_path, random_state):
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    train_path = write_path + 'train.txt'
    test_path = write_path + 'test.txt'
    min_data = load_data(read_path + '/min/all.txt')
    more_data = load_data(read_path + '/more/all.txt')
    train, test = train_test_split(min_data, random_state)
    train.extend(more_data)
    writefile(train, train_path)
    writefile(test, test_path)

def splits_to_more(read_path, write_path, random_state):
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    train_path = write_path + 'train.txt'
    test_path = write_path + 'test.txt'
    min_data = load_data(read_path + '/min/all.txt')
    more_data = load_data(read_path + '/more/all.txt')
    train, test = train_test_split(more_data, random_state)
    train.extend(min_data)
    writefile(train, train_path)
    writefile(test, test_path)
        
def writefile(data, file_path):
    with open(file_path, 'a+') as f:
        for d in data:
            f.write(str(d[0]) + '\t' + str(d[1]) + '\t' + str(d[2]) + '\t' + str(d[3])  + '\t' + str(d[4]) + '\t' + str(d[5]) + '\t' + str(d[6]) + '\t' + str(d[7]) + '\t' + str(d[8]) +'\n')

def load_data(file_path):
    a = []
    with open(file_path, 'r') as f:
        data = f.readlines()
        for line in data:
            d = line.split()
            #yahooo
            a.append((d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]))
            #tripadvisor
            # a.append((d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[9]))            
    return a

def train_test_split(data, random_state):
    n_ratings = len(data)
    test_size, train_size = validate_train_test_sizes(n_ratings)
    rng = np.random.RandomState(random_state)
    permutation = rng.permutation(len(data))
    trainset = [data[i] for i in permutation[:test_size]]
    testset = [data[i] for i in permutation[test_size:(test_size + train_size)]]
    return trainset, testset


def validate_train_test_sizes(n_ratings, test_size = 0.2, train_size = None):
    #
    if test_size is not None and test_size >= n_ratings:
        raise ValueError('test_size={0} should be less than the number of '
                             'ratings {1}'.format(test_size, n_ratings))
    if train_size is not None and train_size >= n_ratings:
        raise ValueError('train_size={0} should be less than the number of'
                             ' ratings {1}'.format(train_size, n_ratings))

    if np.asarray(test_size).dtype.kind == 'f':
        test_size = ceil(test_size * n_ratings)
    if train_size is None:
        train_size = n_ratings - test_size
    elif np.asarray(train_size).dtype.kind == 'f':
        train_size = floor(train_size * n_ratings)

    if test_size is None:
        test_size = n_ratings - train_size

    if train_size + test_size > n_ratings:
        raise ValueError('The sum of train_size and test_size ({0}) '
                             'should be smaller than the number of '
                             'ratings {1}.'.format(train_size + test_size,
                                                   n_ratings))
    return int(train_size), int(test_size)

in_path = 'tripadvisor'
out_path = './tripadvisor(v1)/'     
random_states = 32

for k in range(1,21,1):
    main(k, in_path, out_path, random_states)
