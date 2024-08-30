import pandas as pd
import numpy as np
from pprint import pprint


class TopK:
    def __init__(self, K, df, log_file):
        self.K = K
        self.topk = K
        self.df = df
        self.log_file = log_file

    def TOPK(self, R, T):
        res = {}
        p = 4
        res['PRECISION@' + str(self.K)] = round(self.PRECISION(R, T), p)
        res['RECALL@' + str(self.K)] = round(self.RECALL(R, T), p)
        res['HR@' + str(self.K)] = round(self.HR(R, T), p)
        res['MAP@' + str(self.K)] = round(self.MAP(R, T), p)
        res['MRR@' + str(self.K)] = round(self.MRR(R, T), p)
        res['NDCG@' + str(self.K)] = round(self.NDCG(R, T), p)
        res['AUTC@' + str(self.K)] = round(self.AUTC(), p)
        return res

    def evaluate(self):
        global temp
        temp = self.df[['user_id', 'video_id', 'test_label', 'date_diff', 'pred']]
        temp.columns = ['user_id', 'item_id', 'test_label', 'date_diff', 'pred']
        true_df = temp[temp.test_label == 1].groupby(['user_id']).agg({'item_id': lambda x: \
            list(x)}).reset_index().sort_values(by=['user_id'])
        x = pd.DataFrame(temp[~temp.user_id.isin(true_df['user_id'])]['user_id'].drop_duplicates())
        x['item_id'] = [[] for i in range(len(x))]
        #         true_df = pd.concat([true_df,x])
        temp = temp[temp.user_id.isin(true_df['user_id'])]
        temp = temp.sort_values(by=['user_id', 'pred'], ascending=False)
        rank_df = temp.groupby(['user_id']).agg({'item_id': lambda x: list(x)}).reset_index().sort_values(
            by=['user_id'])
        rank_df['top' + str(self.topk)] = rank_df['item_id'].apply(
            lambda x: x[:self.topk] if len(x) >= self.topk else x)

        assert len(true_df) == len(rank_df)

        df = pd.merge(left=true_df, right=rank_df[['user_id', 'top' + str(self.topk)]], on=['user_id'])

        assert len(df) == len(df.dropna(how='any'))

        T = df['item_id'].tolist()
        R = df['top' + str(self.topk)].tolist()

        res = self.TOPK(R, T)
        print('RECALL@{:d} {:.4f} | MAP@{:d} {:.4f} | NDCG@{:d} {:.4f} | HR@{:d} {:.4f} | MRR@{:d} {:.4f}'
              .format(self.topk, res['RECALL@' + str(self.topk)],
                      self.topk, res['MAP@' + str(self.topk)],
                      self.topk, res['NDCG@' + str(self.topk)],
                      self.topk, res['HR@' + str(self.topk)],
                      self.topk, res['MRR@' + str(self.topk)]),
              file=self.log_file)
        print('RECALL@{:d} {:.4f} | MAP@{:d} {:.4f} | NDCG@{:d} {:.4f} | HR@{:d} {:.4f} | MRR@{:d} {:.4f}'
              .format(self.topk, res['RECALL@' + str(self.topk)],
                      self.topk, res['MAP@' + str(self.topk)],
                      self.topk, res['NDCG@' + str(self.topk)],
                      self.topk, res['HR@' + str(self.topk)],
                      self.topk, res['MRR@' + str(self.topk)]))
        # self.summary(self.test_df)

    def PRECISION(self, R, T):
        assert len(R) == len(T)
        res = 0
        for i in range(len(R)):
            res += len(set(R[i]) & set(T[i])) / len(R[0])
        return res / len(R)

    def RECALL(self, R, T):
        assert len(R) == len(T)
        res = 0
        for i in range(len(R)):
            if len(T[i]) > 0:
                res += len(set(R[i]) & set(T[i])) / len(T[i])
        return res / len(R)

    def HR(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        for i in range(len(R)):
            if len(set(R[i]) & set(T[i])) > 0:
                up += 1
        return up / down

    def MAP(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        for i in range(len(R)):
            temp = 0
            hit = 0
            for j in range(len(R[i])):
                if R[i][j] in T[i]:
                    hit += 1
                    temp += hit / (j + 1)
            if hit > 0:
                up += temp / len(T[i])
        return up / down

    def MRR(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        for i in range(len(R)):
            index = -1
            for j in range(len(R[i])):
                if R[i][j] in T[i]:
                    index = R[i].index(R[i][j])
                    break
            if index != -1:
                up += 1 / (index + 1)
        return up / down

    def dcg(self, hits):
        res = 0
        for i in range(len(hits)):
            res += (hits[i] / np.log2(i + 2))
        return res

    def NDCG(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        for i in range(len(R)):
            hits = []
            for j in range(len(R[i])):
                if R[i][j] in T[i]:
                    hits += [1.0]
                else:
                    hits += [0.0]
            if sum(hits) > 0:
                up += (self.dcg(hits) / (self.dcg([1.0 for i in range(len(T[i]))]) + 1))  # 来自wiki的定义，idcg应该是对目标排序。
        return up / down

    def AUTC(self):
        return 0
        # t = self.test_df.groupby(['user_id']).apply(lambda x: self.play_time(x, self.topk))
        # return t.sum() / len(t)

    def summary(self):
        # datediff_list = list(df['date_diff'].unique())
        # tag_list = list(df['tag_pop'].unique())
        # datediff_list.sort(reverse=False)
        # y_pred_dict = {}
        #
        # for tag in tag_list:
        #     y_pred_dict[tag] = []
        #     for datediff in datediff_list:
        #         df1 = df[df.date_diff == datediff]
        #         y_pred_dict[tag].append(df1['pred'].mean())
        #
        # # for datediff in datediff_list:
        # #     df1 = df[]
        #
        # pprint(y_pred_dict, stream=log_file)
        return

    def get_ndcg(self):
        def get_ndcg_summary(df):
            global temp
            temp = df[['user_id', 'video_id', 'test_label', 'date_diff', 'pred']]
            temp.columns = ['user_id', 'item_id', 'test_label', 'date_diff', 'pred']
            true_df = temp[temp.test_label == 1].groupby(['user_id']).agg({'item_id': lambda x: \
                list(x)}).reset_index().sort_values(by=['user_id'])
            x = pd.DataFrame(temp[~temp.user_id.isin(true_df['user_id'])]['user_id'].drop_duplicates())
            x['item_id'] = [[] for i in range(len(x))]
            #         true_df = pd.concat([true_df,x])
            temp = temp[temp.user_id.isin(true_df['user_id'])]
            temp = temp.sort_values(by=['user_id', 'pred'], ascending=False)
            rank_df = temp.groupby(['user_id']).agg({'item_id': lambda x: list(x)}).reset_index().sort_values(
                by=['user_id'])
            rank_df['top' + str(self.topk)] = rank_df['item_id'].apply(
                lambda x: x[:self.topk] if len(x) >= self.topk else x)

            assert len(true_df) == len(rank_df)

            df = pd.merge(left=true_df, right=rank_df[['user_id', 'top' + str(self.topk)]], on=['user_id'])

            assert len(df) == len(df.dropna(how='any'))

            T = df['item_id'].tolist()
            R = df['top' + str(self.topk)].tolist()
            res = self.TOPK(R, T)

            return res['RECALL@' + str(self.topk)]

        ndcg_list = []
        n_day = len(list(self.df['date_diff'].unique()))
        for i in range(n_day):
            df1 = self.df[self.df.date_diff == i]
            ndcg_list.append(get_ndcg_summary(df1))
        print(ndcg_list)

        return None


