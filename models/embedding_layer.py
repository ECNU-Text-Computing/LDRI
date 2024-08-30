import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EmbeddingLayer(nn.Module):
    def __init__(self, sparse_feature_size, embedding_dim, init=True, init_std=0.1):
        """
        return a list of embeddings
        :param sparse_feature_size:
        :param embedding_dim:
        :param init_std:
        :param init:
        """
        super(EmbeddingLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.sparse_feature_size = sparse_feature_size
        self.init_std = init_std

        self.embeddings = []
        for size in sparse_feature_size:
            self.embeddings.append(nn.Embedding(size, self.embedding_dim))
        self.embeddings = nn.ParameterList(self.embeddings)

        self.embed_dim = self.embedding_dim * len(sparse_feature_size)
        # setattr(self, 'embed_dim', self.embed_dim)

        self.bn = nn.BatchNorm1d(self.embed_dim)

        if init:
            self._init_weights()

    def forward(self, x, cate_length_list=None, pre_train=None):
        embeds = []
        if cate_length_list is None:  # get embeddings of id features
            for i in range(len(self.embeddings)):
                embeds.append(self.embeddings[i](x[:, i].long()))  # [batch_size, embedding_dim]
        else:
            for i in range(len(self.embeddings)):
                embeds.append(self.embeddings[i](
                    x[:, sum(cate_length_list[:i]):sum(cate_length_list[:i + 1])].long()))  # .flatten(1)
                # list([batch_size, len1, dim], [batch_size, len2, dim])
        if pre_train is not None:
            pre_embed = nn.Embedding.from_pretrained(pre_train, freeze=True)(x[:, 1].long())
            embeds.append(pre_embed)

        # embeds = self.bn(embeds.type(torch.bfloat16)).float()
        return embeds

    def _init_weights(self):
        return nn.ParameterList([nn.init.normal_(weight, mean=0, std=self.init_std)
                                 for weight in self.embeddings.parameters()])
