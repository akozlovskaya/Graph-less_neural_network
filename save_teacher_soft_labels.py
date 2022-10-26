import argparse
from utils.data_utils import load_data, GraphDataset, graph_collate_fn
from utils.utils import infer_model

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

path_to_data = './data/'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--path_to_data", type=str, default=path_to_data)

    parser.add_argument("--dataset_name", type=str, default="cora")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nn_max_size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--directory", type=str, default="experiments")

    args = parser.parse_args()
    return args


def save_soft_labels(args):
    output_dir = args.path_to_data + f'{args.directory}/exp_teacher_{args.seed}/'
    path_to_model = output_dir + 'best.pkl'
    assert os.path.exists(path_to_model), "learn model first"
    model = torch.load(path_to_model)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = 'cpu'

    """ Load data """
    graph_dict, attr_matrix, labels, idx_dict = load_data(args.path_to_data + 'cora.npz', 0, do_split=False)

    dataset = GraphDataset(graph_dict, attr_matrix, labels, idx_dict["all"], args.nn_max_size, None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=graph_collate_fn)
    soft_labels = infer_model(model, dataloader, device, "log_preds")
    with open(output_dir + 'soft_labels.npy', 'wb') as f:
        np.save(f, soft_labels)


def main():
    args = get_args()
    save_soft_labels(args)


if __name__ == "__main__":
    main()