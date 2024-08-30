# -*- coding: utf-8 -*-
import sys
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
grandgrandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)


class Loader(object):
    def __init__(self, dataset):
        super(Loader, self).__init__()

        if dataset == 'kuairand_1k':
            self.data_path = grandgrandparent_dir + '/dataset_video/KuaiRand/KuaiRand-1K/data/'
        elif dataset == 'kuairand_pure':
            self.data_path = grandgrandparent_dir + '/dataset_video/KuaiRand/KuaiRand-Pure/data/'

        if not os.path.exists(grandparent_dir + '/dataset/'):
            os.mkdir(grandparent_dir + '/dataset/')
        self.save_path = grandparent_dir + '/dataset/'

        self.dataset = dataset

        # self.data_loader()

    def data_loader(self):
        if self.dataset != 'kuairand_1k' and self.dataset != 'kuairand_pure':
            print('No such dataset please check carefully.')

        global log, video_feat, log_sd1, log_sd2, user_info
        user_feature = (['user_active_degree', 'is_live_streamer', 'is_video_author']
                        + ['follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range']
                        # + ['onehot_feat%d' % i for i in range(18)]
                        )

        if self.dataset == 'kuairand_1k':
            log_sd1 = pd.read_csv(self.data_path + 'log_standard_4_08_to_4_21_1k.csv')
            log_sd2 = pd.read_csv(self.data_path + 'log_standard_4_22_to_5_08_1k.csv')
            video_feat = pd.read_csv(self.data_path + 'video_features_basic_1k.csv')
            user_info = pd.read_csv(self.data_path + 'user_features_1k.csv')

        elif self.dataset == 'kuairand_pure':
            log_sd1 = pd.read_csv(self.data_path + 'log_standard_4_08_to_4_21_pure.csv')
            log_sd2 = pd.read_csv(self.data_path + 'log_standard_4_22_to_5_08_pure.csv')
            video_feat = pd.read_csv(self.data_path + 'video_features_basic_pure.csv')
            user_info = pd.read_csv(self.data_path + 'user_features_pure.csv')

        log = pd.concat([log_sd1, log_sd2], axis=0)
        df = pd.merge(left=log, right=video_feat, how='left', on=['video_id']).dropna(how='any')
        user_info = user_info[['user_id'] + user_feature]
        for feat in user_feature[:6]:
            user_info[feat] = self.reorder_id(user_info, feat)
        df = pd.merge(left=df, right=user_info, how='left', on=['user_id']).dropna(how='any')

        # date_diff  type: int
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['upload_dt'] = pd.to_datetime(df['upload_dt'], format='%Y-%m-%d')
        df['date_diff'] = (df['date'] - df['upload_dt']).dt.days.astype('int32')

        # transform str to numerical categorical variable
        df['video_type'] = df['video_type'].map(
            {label: idx for idx, label in enumerate(np.unique(df['video_type']))})
        df['upload_type'] = df['upload_type'].map(
            {label: idx for idx, label in enumerate(np.unique(df['upload_type']))})
        df['music_type'] = df['music_type'].map(
            {label: idx for idx, label in enumerate(np.unique(df['music_type']))})

        # tag info
        df['tag_list'] = df['tag'].apply(lambda x: str(x).strip().split(','))
        total_ls = self.contain_ls(df['tag_list'].values)
        stat_series = pd.Series(total_ls).value_counts()
        count_info = dict(zip(stat_series.index, stat_series.values))
        df['tag_pop'] = df['tag_list'].apply(lambda x: self.compare_max(x, count_info))
        df['tag_pop'] = df['tag_pop'].apply(lambda x: 999 if pd.isna(x) else x)

        # watch_time and duration
        if self.dataset == 'kuairand_1k':
            df1 = pd.DataFrame()
            df1['play_time_ms'] = df['play_time_ms']
            df1['duration_ms'] = df['duration_ms']
            df1['play_time_truncate'] = df1.apply(
                lambda row: row['play_time_ms'] if row['play_time_ms'] < row['duration_ms'] else row['duration_ms'], axis=1)
            df['play_time_truncate'] = df1['play_time_truncate']

        elif self.dataset == 'kuairand_pure':
            df['play_time_truncate'] = df.apply(
                lambda row: row['play_time_ms'] if row['play_time_ms'] < row['duration_ms'] else row['duration_ms'], axis=1)

        df['duration'] = df['duration_ms'].apply(lambda x: np.round(x / 1000))
        df['play_time_truncate'] = df['play_time_truncate'].apply(lambda x: np.round(x / 1000))

        df['watch_ratio'] = df['play_time_ms'] / df['duration_ms']
        df['finish_playing'] = (df['watch_ratio'] >= 1.0).astype('int32')
        # df['binary_pcr'] = self.get_binary_pcr(df)

        df['test_label'] = (df['is_like'] |
                            df['is_comment'] |
                            df['is_follow'] |
                            df['is_forward'] |
                            df['is_profile_enter']).astype('int32')

        df['video_id'] = self.reorder_id(df, 'video_id')
        df['music_id'] = self.reorder_id(df, 'music_id')
        df['tag_pop'] = self.reorder_id(df, 'tag_pop')

        df.to_csv(self.save_path + self.dataset + '.csv', index=False)

        # df.to_csv(self.save_path + self.dataset + '.csv', index=False)

    def reorder_id(self, x, id):
        enc = OrdinalEncoder()
        x[id] = enc.fit_transform(x[id].to_numpy().reshape(-1, 1)).astype('int32')  # 任意行，1列
        return x[id]

    def contain_ls(self, list):
        result_ls = []
        for x in list:
            result_ls.extend(x)
        return result_ls

    def compare_max(self, cat_ls, frac_dict):
        frac_ls = np.array([frac_dict[c] for c in cat_ls])
        cat_ls = np.array(cat_ls)
        frac_sort_cat_ls = cat_ls[np.argsort(frac_ls)][::-1]
        return frac_sort_cat_ls[0]



if __name__ == '__main__':
    Loader('kuairand_1k').data_loader()

