from torch.utils.data import random_split

def split_dataset(dataset, num):
    '''
    :param dataset: the torch.utils.data - dataset
    :param num: number of users/ clients
    :return: [dataset_list], [corresponding type]
    '''
    type_list = []
    dataset_list = random_split(dataset, [len(dataset) / num] * num)
    for d in dataset_list:
        types = []
        for sample in d.data:
            types.append(sample['target'])
        type_list.append(types)
    return dataset_list, type_list