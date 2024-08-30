import torch
import torch.nn as nn
import torch.nn.functional as F
from deepctr_torch.layers.interaction import BiInteractionPooling

from .embedding_layer import *
from .base_model import BaseModel
import warnings

warnings.filterwarnings("ignore")


class NFM(BaseModel):
    def __init__(self,
                 feature_size_map,
                 embed_dim_dnn=16, hidden_dims=[16, 8],
                 dropout=[0.5, 0.5],
                 use_dense=True):
        # super(NFM, self).__init__(feature_size_map=feature_size_map,
        #                           embed_dim_time=16,
        #                           embed_dim_sparse=embed_dim_dnn,
        #                           embed_dim_dense=embed_dim_dnn,
        #                           use_dense=use_dense
        #                           )
        super().__init__(feature_size_map=feature_size_map,
                         use_dense=use_dense)

        self.embed_dim_dnn = embed_dim_dnn
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        self.feature_size_map = feature_size_map
        self.sparse_feature_size = feature_size_map['id_feature'] + feature_size_map['cate_feature']
        self.sparse_field_dim = len(self.sparse_feature_size)
        self.dense_field_dim = feature_size_map['dense_feature'][0] if use_dense else 0
        self.use_dense = use_dense

        # initialize linear part
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        self.fm_embedding = EmbeddingLayer(self.sparse_feature_size, 1, init=True, init_std=0.01)
        self.fm_linear_dense = nn.Linear(self.dense_field_dim, 1)
        self.batch_norm_dense = nn.BatchNorm1d(self.dense_field_dim)

        # initialize dnn part
        self.dnn_embeding = EmbeddingLayer(self.sparse_feature_size, self.embed_dim_dnn)
        self.dnn_linear_dense = nn.Linear(self.dense_field_dim, self.embed_dim_dnn)

        self.BIP = BiInteractionPooling()

        self.batch_norm_dnn = nn.BatchNorm1d(self.embed_dim_dnn)

        self.dnn_out = nn.Sequential(nn.Linear(self.embed_dim_dnn, self.hidden_dims[0]),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout[0]),
                                     nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout[1]),
                                     nn.Linear(self.hidden_dims[-1], 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """FM part"""
        fm_embeds_sparse = self.fm_embedding(x[:, :self.sparse_field_dim].long())
        fm_embeds_sparse = torch.stack(fm_embeds_sparse, dim=1)
        fm_embeds_sparse = torch.sum(fm_embeds_sparse, dim=1, keepdim=False)

        if self.use_dense:
            dense = self.batch_norm_dense(x[:, -self.dense_field_dim:].type(torch.float32)).float()
            # dense = x[:, -self.dense_field_dim:].type(torch.float32)
            dense = torch.where(torch.isnan(dense), torch.zeros_like(dense), dense)
            fm_embeds_dense = self.fm_linear_dense(dense).float()  # [batch_size, 1]
            fm_out = self.bias + fm_embeds_sparse + fm_embeds_dense.float()  # [batch_size, 1]
        else:
            fm_out = self.bias + fm_embeds_sparse

        """DNN part"""
        dnn_embeds_sparse = torch.stack(self.dnn_embeding(x[:, :self.sparse_field_dim].long()),
                                        dim=1)

        dnn_embeds = dnn_embeds_sparse
        dnn_hidden = self.BIP(dnn_embeds).squeeze()  # [batch_size, 1, dim] -> [batch_size, dim]
        dnn_hidden = self.batch_norm_dnn(dnn_hidden)
        dnn_out = self.dnn_out(dnn_hidden)

        out = self.sigmoid(fm_out + dnn_out)

        return out
