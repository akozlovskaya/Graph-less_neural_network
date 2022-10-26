from models.mlp import MLP
import torch
from torch import nn


class SAGE(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_p=0.,
                 ):
        super(SAGE, self).__init__()
        self.do = nn.Dropout(p=dropout_p)
        self.transform = nn.Linear(2 * input_dim, output_dim)

    def forward(self, h, h_nn):
        h_nn_aggr = torch.sum(h_nn, dim=1)
        h_out = self.transform(torch.concat([h, h_nn_aggr], dim=-1))
        return h_out


class GraphModel(nn.Module):
    def __init__(self,
                graph_args,
                mlp_args):
        super().__init__()

        self.graph_model = SAGE(**graph_args)
        self.mlp_model = MLP(**mlp_args)

    def forward(self, x):
        h_graph = self.graph_model(x['h'], x['h_nn'])
        mlp_out = self.mlp_model({'h': h_graph})

        return { 'h_graph': h_graph, **mlp_out }
