import argparse
from utils.data_utils import load_data, load_numpy_arr,\
    LinearDataset, node_collate_fn,\
    GraphDataset, graph_collate_fn,\
    StudentLinearDataset, node_student_collate_fn

from models.mlp import MLP
from models.gnn import GraphModel
from utils.losses import TeacherLoss, StudentLoss

from utils.utils import fit_model, set_seed

import os
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

path_to_data = './data/'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--path_to_data", type=str, default=path_to_data)

    parser.add_argument("--dataset_name", type=str, default="cora")
    parser.add_argument("--mode", type=str, default="teacher")

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout_p", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--nn_max_size", type=int, default=15)
    parser.add_argument("--directory", type=str, default="experiments")
    parser.add_argument("--weight_decay", type=float, default=0.0)

    args = parser.parse_args()
    return args


def run_experiment(args):

    set_seed(args.seed)
    
    output_dir = args.path_to_data + f'{args.directory}/exp_{args.mode}_{str(args.seed)}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print('already done')

    """ Load data """
    graph_dict, attr_matrix, labels, idx_dict = load_data(args.path_to_data + 'cora.npz', args.seed)
    path_to_soft_labels = args.path_to_data  + f'{args.directory}/exp_teacher_{args.seed}/soft_labels.npy'
    soft_labels = load_numpy_arr(path_to_soft_labels)
    label_size = np.max(labels) + 1

    if args.mode == "teacher":
        Dataset = GraphDataset
        collate_fn = graph_collate_fn
        criterion = TeacherLoss()
        model = GraphModel(graph_args={"input_dim": attr_matrix.shape[1], "output_dim": args.hidden_dim},
                       mlp_args={"input_dim": args.hidden_dim, "output_dim": label_size},
                       )
    elif args.mode == "mlp":
        Dataset = LinearDataset
        collate_fn = node_collate_fn
        criterion = TeacherLoss()
        model = MLP(
            input_dim=attr_matrix.shape[1], output_dim=label_size,
            num_layers=args.num_layers, dropout_p=args.dropout_p
        )
    elif args.mode == "student_mlp":
        Dataset = StudentLinearDataset
        collate_fn = node_student_collate_fn
        criterion = StudentLoss(0.0)
        model = MLP(input_dim=attr_matrix.shape[1], output_dim=label_size)
    else:
        assert "Your mode is not implemented"

    names = ['train', 'valid', 'test']
    dataloaders = dict()
    for name in names:
        dataset = Dataset(graph_dict, attr_matrix, labels, idx_dict[name], args.nn_max_size, soft_labels)
        dataloaders[name] = DataLoader(dataset, batch_size=args.batch_size,
                                       shuffle=name=='train', collate_fn=collate_fn)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fit_model(model, dataloaders, criterion, optim, args, output_dir)


def main():
    args = get_args()
    run_experiment(args)


if __name__ == "__main__":
    main()