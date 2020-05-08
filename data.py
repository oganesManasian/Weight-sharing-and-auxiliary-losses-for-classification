import torch
from torch.utils.data import DataLoader
from collections import Counter

from dlc_practical_prologue import generate_pair_sets


def balance_data_classes(train_input, train_class, train_digit, mode="oversampling"):
    """Balances data by over or under sampling"""
    count_train = Counter([el.item() for el in train_class])
    diff = count_train[1] - count_train[0]
    if diff == 0:  # No balancing needed
        return train_input, train_class, train_digit

    major_class, minor_class = (1, 0) if diff > 0 else (0, 1)
    major_class_indices = [ind for ind in range(len(train_class)) if train_class[ind].item() == major_class]
    minor_class_indices = [ind for ind in range(len(train_class)) if train_class[ind].item() == minor_class]

    # Build new indices
    if mode == "oversampling":
        perm = torch.randperm(len(minor_class_indices))[:diff]
        indices_to_add = [minor_class_indices[ind] for ind in perm]
        minor_class_indices += indices_to_add  # Adding random instances
    elif mode == "undersampling":
        major_class_indices = major_class_indices[:-diff]  # Deleting some of instances
    else:
        raise NotImplementedError

    # Concatenate and permute indices
    new_data_indices = major_class_indices + minor_class_indices
    perm = torch.randperm(len(new_data_indices))
    new_data_indices = [new_data_indices[ind] for ind in perm]

    # Build new data
    train_input = train_input[new_data_indices]
    train_class = train_class[new_data_indices]
    train_digit = train_digit[new_data_indices]
    return train_input, train_class, train_digit


def get_data_generator(N=1000, mode=None):
    """
    Creates function for generating data
    :param N: number of samples in data
    :param mode: mode for data preprocessing (none, oversampling, undersampling)
    :return: function for generating data
    """
    def generate_data(N=N, mode=mode):
        """
        Generates pair datasets and returns them as DataLoader classes
        :param N: number of samples in data
        :param mode: mode for data preprocessing (none, oversampling, undersampling)
        :return:
        """
        train_input, train_class, train_digit, test_input, test_class, test_digit = generate_pair_sets(N)

        # Normalising data
        mean = torch.mean(train_input)
        std = torch.std(train_input)
        train_input = (train_input - mean) / std
        test_input = (test_input - mean) / std

        # Balancing data
        if mode:
            train_input, train_class, train_digit = balance_data_classes(train_input, train_class, train_digit, mode)

        train_loader = DataLoader(list(zip(train_input, train_class, train_digit)), batch_size=64)
        test_loader = DataLoader(list(zip(test_input, test_class, test_digit)), batch_size=64)
        return train_loader, test_loader

    return generate_data
