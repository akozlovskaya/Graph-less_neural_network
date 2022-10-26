import random
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
import scipy.sparse as sp


def get_train_val_test_split(
    random_state,
    num_samples,
    train_size=0.8,
    val_size=0.1,
):

    remaining_indices = list(range(num_samples))
    train_indices = random_state.choice(
        remaining_indices, int(train_size * num_samples), replace=False
    )

    remaining_indices = np.setdiff1d(remaining_indices, train_indices)
    val_indices = random_state.choice(remaining_indices, int(val_size * num_samples), replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    return train_indices, val_indices, test_indices


def load_data(file_name, seed, do_split=True):
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        graph_dict = defaultdict(list)
        for i in range(loader["adj_shape"][0]):
            for ind in range(loader['adj_indptr'][i], loader['adj_indptr'][i + 1]):
                graph_dict[i].append(loader["adj_indices"][ind])

        attr_matrix = sp.csr_matrix(
            (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
            shape=loader["attr_shape"],
        ).toarray()

        labels = loader["labels"]
    num_elems = labels.shape[0]
    if do_split:
        random_state = np.random.RandomState(seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, num_elems)
        split_dict = {'train': idx_train, 'valid': idx_val, 'test': idx_test}
    else:
        split_dict = {"all": range(num_elems)}

    return graph_dict, attr_matrix, labels, split_dict


def load_numpy_arr(path_to_read):
    arr = None
    if os.path.exists(path_to_read):
        with open(path_to_read, 'rb') as f:
            arr = np.load(f)
    return arr


class LinearDataset(Dataset):
    def __init__(self,
                 graph_dict,
                 attr_matrix,
                 labels,
                 idxs,
                 fake_arg,
                 fake_arg_2,
                 ):
        self.node_features = attr_matrix
        self.idxs = idxs
        self.labels = labels

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        idx = self.idxs[item]

        return {'labels': self.labels[idx],
                'node_features': self.node_features[idx]
                }


class GraphDataset(Dataset):
    def __init__(self,
                 graph_dict,
                 attr_matrix,
                 labels,
                 idxs,
                 nn_max_size,
                 fake_arg,
                 ):
        self.node_features = attr_matrix
        self.idxs = idxs
        self.labels = labels

        self.graph_dict = graph_dict
        self.nn_max_size = nn_max_size
        self.emb_size = self.node_features.shape[1]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        idx = self.idxs[item]
        nn_idxs = self.graph_dict[idx]
        random.shuffle(nn_idxs)
        nn_features = []
        nn_size = len(nn_idxs)
        if len(nn_idxs) > 0:
            for nn_idx in nn_idxs[:self.nn_max_size]:
                nn_features.append(self.node_features[nn_idx])
            nn_features = np.concatenate(nn_features)
        else:
            nn_features = np.zeros((self.emb_size))
            nn_size += 1

        nn_features = nn_features.reshape((self.emb_size, -1))

        return {'labels': self.labels[idx],
                'node_features': self.node_features[idx],
                'nn_features': nn_features,
                'nn_size': nn_size
                }


def node_collate_fn(batch):
    embs = []
    labels = []
    for el in batch:
        labels.append(el['labels'])
        embs.append(el['node_features'])
    embs = torch.FloatTensor(np.array(embs))
    labels = torch.LongTensor(labels)

    return {'labels': labels, 'h': embs}


def graph_collate_fn(batch):
    embs = []
    labels = []
    nn_embs = []
    nn_lenghts = []
    for el in batch:
        labels.append(el['labels'])
        embs.append(el['node_features'])
        nn_lenghts.append(el['nn_size'])
        nn_embs.append(torch.FloatTensor(np.array(el['nn_features'])).transpose(0, 1))
    embs = torch.FloatTensor(np.array(embs))
    labels = torch.LongTensor(labels)
    nn_lenghts = torch.LongTensor(nn_lenghts)

    nn_embs = pad_sequence(nn_embs, batch_first=True)

    return {'labels': labels, 'h': embs, 'h_nn': nn_embs, 'nn_lenghts': nn_lenghts}


class StudentLinearDataset(Dataset):
    def __init__(self,
                 graph_dict,
                 attr_matrix,
                 labels,
                 idxs,
                 fake_arg,
                 teacher_soft_labels,
                 ):
        self.node_features = attr_matrix
        self.idxs = idxs
        self.labels = labels
        self.teacher_soft_labels = teacher_soft_labels

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        idx = self.idxs[item]

        return {'labels': self.labels[idx],
                'node_features': self.node_features[idx],
                'teacher_soft_labels': self.teacher_soft_labels[idx],
                }


def node_student_collate_fn(batch):
    embs = []
    labels = []
    soft_labels = []
    for el in batch:
        labels.append(el['labels'])
        embs.append(el['node_features'])
        soft_labels.append(el['teacher_soft_labels'])
    embs = torch.FloatTensor(np.array(embs))
    labels = torch.LongTensor(labels)
    soft_labels = torch.FloatTensor(np.array(soft_labels))

    return {'labels': labels, 'h': embs, 'teacher_soft_labels': soft_labels}