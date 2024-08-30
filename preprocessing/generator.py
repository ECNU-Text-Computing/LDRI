# -*- coding: utf-8 -*-
from datetime import date

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from .loader import *
import sys

sys.path.append('..')


class Generator(Loader):
    def __init__(self, dataset, use_dense=True):
        super(Generator, self).__init__(dataset)

        self.use_dense = use_dense
        print('Dataset:', self.dataset)

        # selected feats of user and item
        self.id_feature_user = ['user_id']
        self.cate_feature_user = ['user_active_degree', 'is_live_streamer', 'is_video_author',
                                  'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range']
        self.sparse_feature_user = self.id_feature_user + self.cate_feature_user
        self.dense_feature_user = []

        self.id_feature_item = ['music_id', 'video_id']
        self.cate_feature_item = ['tag_pop', 'video_type', 'upload_type', 'music_type']
        self.sparse_feature_item = self.id_feature_item + self.cate_feature_item
        self.dense_feature_item = ['watch_ratio', 'play_time_ms', 'comment_stay_time', 'profile_stay_time'] \
            if self.use_dense else []
        self.dense_field_dim_item = [len(self.dense_feature_item)] if self.use_dense else []

        self.id_feature = self.id_feature_user + self.id_feature_item
        self.cate_feature = self.cate_feature_user + self.cate_feature_item
        self.sparse_feature = self.sparse_feature_user + self.sparse_feature_item
        self.dense_feature = self.dense_feature_user + self.dense_feature_item
        self.dense_field_dim = [len(self.dense_feature)] if self.use_dense else []

        self.feature = self.sparse_feature + self.dense_feature
        self.feature_map = dict({'id_feature': self.id_feature,
                                 'cate_feature': self.cate_feature,
                                 'dense_feature': self.dense_feature})
        self.n_day = 30

    def wrapper(self, batch_size=512, num_samples=None):
        # load preprocessed dataframe
        global train_df, test_df, train_loader, test_loader
        log_df = pd.read_csv(self.save_path + self.dataset + '.csv')
        log_df = log_df.fillna(value=0.0)
        log_df['date_diff'] = log_df['date_diff'].apply(lambda x: min(x, self.n_day - 1))


        confounder_info = []
        for day in range(self.n_day):
            log_df1 = log_df[log_df.date_diff == day]
            confounder_info.append(len(log_df1) / len(log_df))
        confounder_info = torch.from_numpy(np.array(confounder_info))

        # set test label
        train_label = 'test_label'
        test_label = 'test_label'

        # get statistics
        self.get_statistics(log_df)

        # split data by interaction date
        log_df['date'] = pd.to_datetime(log_df['date'], format='%Y-%m-%d').dt.date
        if self.dataset == 'kuairand_1k':
            train_df = log_df[(log_df['date'] >= date(2022, 4, 8)) & (log_df['date'] <= date(2022, 4, 28))]
            valid_df = log_df[(log_df['date'] >= date(2022, 4, 29)) & (log_df['date'] <= date(2022, 5, 1))]
            test_df = log_df[(log_df['date'] >= date(2022, 5, 2)) & (log_df['date'] <= date(2022, 5, 8))]
            print('Length of logs:', len(log_df))

        elif self.dataset == 'kuairand_pure':
            train_df, test_df = train_test_split(log_df, test_size=0.4)
            valid_df = test_df[:int(len(test_df) * 0.25)]
            test_df = test_df[int(len(test_df) * 0.25):]

        if num_samples is not None:
            train_df = train_df.iloc[:int(num_samples * 0.6), :]
            test_df = test_df.iloc[:int(num_samples * 0.4), :]
            shuffle = True
        else:
            shuffle = False

        train_x = np.concatenate([np.array(train_df[self.id_feature]),
                                  np.array(train_df[self.cate_feature]),
                                  np.array(train_df[self.dense_feature])], axis=1)
        valid_x = np.concatenate([np.array(valid_df[self.id_feature]),
                                  np.array(valid_df[self.cate_feature]),
                                  np.array(valid_df[self.dense_feature])], axis=1)
        test_x = np.concatenate([np.array(test_df[self.id_feature]),
                                 np.array(test_df[self.cate_feature]),
                                 np.array(test_df[self.dense_feature])], axis=1)

        # wrap loaders
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x),
                          torch.from_numpy(np.array(train_df['date_diff'])),
                          torch.from_numpy(np.array(train_df[train_label])).float().unsqueeze(1)),
            shuffle=shuffle, batch_size=batch_size, num_workers=8, pin_memory=True, prefetch_factor=2)
        valid_loader = DataLoader(
            TensorDataset(torch.from_numpy(valid_x),
                          torch.from_numpy(np.array(valid_df['date_diff'])),
                          torch.from_numpy(np.array(valid_df[valid_label])).float().unsqueeze(1)),
            shuffle=shuffle, batch_size=batch_size, num_workers=8, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_x),
                          torch.from_numpy(np.array(test_df['date_diff'])),
                          torch.from_numpy(np.array(test_df[test_label])).float().unsqueeze(1)),
            shuffle=shuffle, batch_size=batch_size, num_workers=8, pin_memory=True, prefetch_factor=2)
        return train_loader, valid_loader, test_loader, train_df, valid_df, test_df, confounder_info


    def get_statistics(self, data_df):
        id_feature_size = [int(max(data_df[i]) + 1) for i in self.id_feature]
        cate_feature_size = [int(max(data_df[i]) + 1) for i in self.cate_feature]

        feature_size_map = dict({'id_feature': id_feature_size,
                                 'cate_feature': cate_feature_size,
                                 'dense_feature': self.dense_field_dim})
        setattr(Generator, 'feature_size_map', feature_size_map)

        feature_size_map_item = dict(
            {'id_feature_item': [int(max(data_df[i]) + 1) for i in self.id_feature_item],
             'cate_feature_item': [int(max(data_df[i]) + 1) for i in self.cate_feature_item],
             'dense_feature_item': self.dense_field_dim_item
             })
        setattr(Generator, 'feature_size_map_item', feature_size_map_item)


if __name__ == '__main__':
    Generator('kuairand_pure').wrapper()
