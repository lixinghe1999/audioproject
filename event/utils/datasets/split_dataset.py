import torch
from torch.utils.data import Subset
import numpy as np
import random
def find_base_class_random(class_idx_to_label, type_list, num, group):
    types = list(class_idx_to_label.keys())
    tmp = []
    for ts in type_list:
        for i in range(num):
            remove_types = [x for x in types if x not in ts]
            random_type = random.sample(remove_types, group) + ts
            tmp.append(random_type)
    return tmp

def split_type(class_idx_to_label, num):
    types = list(class_idx_to_label.keys())
    types_list = []
    for i in range(num):
        types_list.append(types[i * int(len(types) / num): (i + 1) * int(len(types) / num)])
    return types_list
def split_type_random(class_idx_to_label, num, group):
    types = list(class_idx_to_label.keys())
    types_list = []
    for i in range(num):
        random_type = random.sample(types, group)
        types_list.append(random_type)
    return types_list
def split_dataset(dataset, num):
    '''
    :param dataset: the torch.utils.data - dataset
    :param num: number of users/ clients
    :return: [dataset_list], [corresponding type]
    '''
    type_list = []
    dataset_list = []
    # random_index = torch.randperm(len(dataset))
    random_index = torch.arange(len(dataset))
    for i in range(num):
        sel_index = random_index[i * int(len(dataset) / num): (i+1) * int(len(dataset) / num)]
        d = Subset(dataset, sel_index)
        dataset_list.append(d)
        types = []
        for i in sel_index:
            sample = dataset.data[i]
            types.append(sample['target'])
        type_list.append(set(types))
    return dataset_list, type_list

def split_dataset_type(dataset, type_list):
    '''
    :param dataset: the torch.utils.data - dataset
    :param type_list: the corresponding type for each dataset
    :return: dataset_list
    '''
    type_dict = dict()
    for index, sample in enumerate(dataset.data):
        target = sample['target']
        if target in type_dict:
            type_dict[target].append(index)
        else:
            type_dict[target] = [index]
    dataset_list = []
    for ts in type_list:
        subset_slice = []
        for t in ts:
            subset_slice += type_dict[t]
        d = Subset(dataset, subset_slice)
        dataset_list.append(d)
    return dataset_list

