from torch.utils.data import random_split, Subset

def split_dataset(dataset, num):
    '''
    :param dataset: the torch.utils.data - dataset
    :param num: number of users/ clients
    :return: [dataset_list], [corresponding type]
    '''
    type_list = []
    dataset_list = []
    for i in range(num):
        dataset = Subset(dataset, [i, (i+1) * int(len(dataset) / num)])
        dataset_list.append(dataset)
        types = []
        for sample in dataset.data:
            types.append(sample['target'])
        type_list.append(types)
    return dataset_list, type_list