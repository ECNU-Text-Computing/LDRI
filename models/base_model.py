import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self,
                 feature_size_map,
                 embed_dim_sparse=16,
                 use_dense=True):
        super().__init__()

        self.use_dense = use_dense

        self.feature_size_map = feature_size_map
        self.sparse_feature_size = feature_size_map['id_feature'] + feature_size_map['cate_feature']
        self.sparse_field_dim = len(self.sparse_feature_size)
        self.dense_field_dim = feature_size_map['dense_feature'][0] if use_dense else 0

        self.embed_dim_sparse = embed_dim_sparse

        self.embed_dim_cat = self.embed_dim_sparse * self.sparse_field_dim + \
                             self.dense_field_dim

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
