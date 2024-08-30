import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding_layer import EmbeddingLayer
from .base_model import BaseModel
from .NFM import NFM


class LDRI(nn.Module):
    def __init__(self,
                 device,
                 feature_size_map,
                 feature_size_map_item,
                 n_day, n=1,
                 embed_dim_sparse_item=16,
                 use_dense=True
                 ):
        super().__init__()

        self.device = device
        self.n_day = n_day
        self.n = int(n)
        self.use_dense = use_dense

        self.backbone = NFM(feature_size_map=feature_size_map,
                            embed_dim_dnn=16,
                            hidden_dims=[16, 8],
                            dropout=[0.5, 0.5],
                            use_dense=self.use_dense)


        self.feature_size_map_item = feature_size_map_item
        self.sparse_feature_size_item = (feature_size_map_item['id_feature_item'] +
                                         feature_size_map_item['cate_feature_item'])
        self.sparse_field_dim_item = len(self.sparse_feature_size_item)
        self.dense_field_dim_item = feature_size_map_item['dense_feature_item'][0] if use_dense else 0
        self.embed_dim_sparse_item = embed_dim_sparse_item
        self.embed_dim_cat_item = self.embed_dim_sparse_item * self.sparse_field_dim_item + \
                                  self.dense_field_dim_item

        self.embedding = EmbeddingLayer(self.sparse_feature_size_item, self.embed_dim_sparse_item, init=True, init_std=0.01)
        self.batch_norm_dense = nn.BatchNorm1d(self.dense_field_dim_item)
        self.perceptron = TimeSensPerceptron(device=self.device,
                                             embed_dim=self.embed_dim_cat_item,
                                             n_day=self.n_day,
                                             n=self.n)

    def forward(self, x, diff):
        ym = self.backbone(x)

        x_item = x[:, [1, 2, 9, 10, 11, 12]]  # filter feat that are only about item
        embed_sparse = torch.cat(self.embedding(x_item[:, :self.sparse_field_dim_item].long()),
                                 dim=1)
        embed = embed_sparse

        pcp_x = embed
        yt, yt_mat = self.perceptron(pcp_x, diff)

        return ym, yt, yt_mat  # (bs, 1), (bs, 30)


class TimeSensPerceptron(nn.Module):
    def __init__(self, device, embed_dim, n_day, n=1):
        super().__init__()

        self.device = device
        self.embed_dim = embed_dim
        self.n_day = n_day
        self.n = n

        self.hidden_dim = [128, 64, 32]

        self.proj = nn.Sequential(nn.Linear(self.embed_dim, 64),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(64, self.n_day),
                                  nn.ReLU())

        self.proj_second = nn.Linear(self.n_day, self.n_day)
        self.relu = nn.ReLU()

    def forward(self, x, diff):
        hidden = self.proj(x)
        hidden_orig = hidden.clone()
        hidden = self.proj_second(hidden)
        hidden = hidden + hidden_orig

        out = hidden[torch.arange(diff.shape[0]), diff].float()

        for i in range(1, self.n + 1):
            left_idx = torch.clamp(diff - i, min=0)
            right_idx = torch.clamp(diff + i, max=self.n_day - 1)
            out += hidden[torch.arange(diff.shape[0]),
                          left_idx].float()
            out += hidden[torch.arange(diff.shape[0]),
                          right_idx].float()

        out = out / (2 * self.n + 1)

        out = nn.Sigmoid()(out)
        return out.view(-1, 1), hidden
