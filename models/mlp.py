from torch import nn


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=1,
                 num_layers=1,
                 dropout_p=0.,
                 use_norm=False,
                 # task='classification'
                 ):
        super(MLP, self).__init__()
        assert hidden_dim > 0, "Set up correct hidden dimension"
        assert num_layers > 0, "Need at least 1 layers"
        self.dims = [input_dim] + (num_layers - 1) * [hidden_dim] + [output_dim]
        self.dropout = nn.Dropout(p=dropout_p)

        self.log_softmax = nn.LogSoftmax(dim=-1)

        # print("mlp  ", input_dim, output_dim)
        self.layers = nn.ModuleList()
        self.norm = nn.Identity()

        if use_norm:
            self.norm = nn.BatchNorm1d(self.dims[0])

        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(self.dropout)

    def forward(self, x):
        h_list = []
        h = self.norm(x['h'])
        last_hidden = None
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h_list.append(h)

        log_preds = self.log_softmax(h)

        return {"logits": h, "log_preds": log_preds}
