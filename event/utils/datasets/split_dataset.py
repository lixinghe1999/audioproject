import torch
from torch.utils.data import random_split, Subset
import numpy as np
def split_dataset(dataset, num):
    '''
    :param dataset: the torch.utils.data - dataset
    :param num: number of users/ clients
    :return: [dataset_list], [corresponding type]
    '''
    type_list = []
    dataset_list = []
    random_index = torch.randperm(len(dataset), dtype=int)
    for i in range(num):
        sel_index = random_index[i * int(len(dataset) / num): (i+1) * int(len(dataset) / num)]
        print(sel_index)
        d = Subset(dataset, sel_index)
        dataset_list.append(d)
        types = []
        for sample in dataset.data[sel_index]:
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

