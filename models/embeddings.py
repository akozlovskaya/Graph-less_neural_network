import torch
from torch import nn

class Embedding_module(nn.Module):

    def __init__(self, input_dim=-1, dict_size=2, emb_size=1):
        super(Embedding_module, self).__init__()
        self.emb = nn.Embedding(input_dim, output_dim)

    def forward(self, x):
        half_1, half_2 = x.int().chunk(2, dim=1)
        shape = half_1.shape
        half_1_embs = self.emb(half_1).reshape(shape)
        h_concated = torch.concat([half_1_embs, half_2], dim=1)
        return h_concated
       
class Embeddings_module(nn.Module):

    def __init__(self, input_dim, dict_size=2, emb_size=1):
        super(Embeddings_module, self).__init__()
        self.emb = nn.ModuleList(
            [ nn.Embedding(dict_size, emb_size) for i in range((input_dim + 1) // 2)]
        )

    def forward(self, x):
        half_1, half_2 = x.int().chunk(2, dim=1)
        h_concated = torch.concat(
            [ e(v) for e, v in zip (self.emb, half_1.T)] + [half_2],
            dim=-1
        )
        return h_concated