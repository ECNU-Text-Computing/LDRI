import pandas as pd
import numpy as np
from time import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from .metric import TopK
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, parameter_count_table

import warnings
import os

warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


class TrainAndTest(object):
    def __init__(self, model, backbone, device,
                 optimizer, seed,
                 train_loader, valid_loader, test_loader, train_df, valid_df, test_df,
                 confounder_info,
                 dataset, num_samples, epochs,
                 is_train=True, is_valid=True, load_epoch=32,
                 n=1, alpha=0.5, beta=0.5):

        self.model = model
        self.backbone = backbone
        self.device = device
        self.model.to(self.device)

        self.optimizer = optimizer

        self.epochs = epochs
        self.dataset = dataset
        self.num_samples = num_samples

        self.is_train = is_train
        self.is_valid = is_valid
        self.load_epoch = load_epoch

        self.save_path = os.path.join(parent_dir, 'logs')
        self.log_file = open(os.path.join(self.save_path,
                                          f'{self.model_name}_{self.backbone}_{self.dataset}.txt'), 'a')
        self.save_path_pth = os.path.join(self.save_path, 'pth_files')

        self.save_name_pth = os.path.join(self.save_path_pth,
                                          f'{self.model_name}_{self.backbone}_{self.dataset}_{self.num_samples}')
        print(self.save_name_pth)

        print('\n\n',
              '\nTime:', str(datetime.now()),
              '\nSeed:', seed,
              '\nTrain or Test:', 'Train' if self.is_train else 'Test',
              '\nNum of samples:', self.num_samples if self.num_samples is not None else 'full',
              '\nNum of epochs:', self.epochs if self.is_train else None,
              '\nPrediction mode:', 'do-calculus, w/o do'
              '\nN:', n,
              '\nAlpha:', alpha,
              '\nBeta(OURS):', beta,
              '\n',
              file=self.log_file
              )

        if is_train > 0:
            self.train(self.model, alpha, beta, train_loader, valid_loader, train_df, valid_df, confounder_info)
        else:
            if self.model_name == 'OURS':
                self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='w/o_ba', beta=beta)
                self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='do', beta=beta)


    def train(self, model, alpha, beta, train_loader, valid_loader, train_df, valid_df, confounder_info):
        global train_loss, bce_loss_ym, bce_loss_yt, valid_loss
        optimizer = self.optimizer

        # model.load_state_dict(torch.load(self.save_name_pth + '_epoch64.pth'))
        model.train()
        print(f'Start training the {self.model_name} with {self.backbone} on {self.dataset} dataset...')
        start_time = time()
        min_loss = 2.0

        for epoch in tqdm(range(self.epochs)):
            for idx, (x, diff, y) in enumerate(train_loader):
                if self.model_name == 'OURS':
                    ym, yt, yt_mat = model.forward(x.to(self.device), diff.to(self.device))
                    bce_loss_ym = nn.BCELoss()(ym, y.to(self.device))
                    bce_loss_yt = nn.BCELoss()(yt, y.to(self.device))
                    train_loss = alpha * bce_loss_ym + (1 - alpha) * bce_loss_yt
                elif self.model_name == 'TaFR':
                    ym = model.forward(x.to(self.device), diff.to(self.device))
                    bce_loss_ym = nn.BCELoss()(ym, y.to(self.device))
                    bce_loss_yt = 0.
                    train_loss = bce_loss_ym


                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            if (epoch + 1) % 2 == 0 or (epoch + 1) == self.epochs:
                if self.model_name == 'OURS':
                    valid_loss = self.valid_and_test(model, valid_loader, valid_df, confounder_info, pred_mode='w/o_ba',
                                                     beta=beta)
                    valid_loss = self.valid_and_test(model, valid_loader, valid_df, confounder_info, pred_mode='do',
                                                     beta=beta)

                print(
                    'epoch {:04d}/{:04d} | ym loss {:.4f} | yt loss {:.4f} | valid loss {:.4f} | time {:.4f} \n'
                    .format(epoch + 1, self.epochs, bce_loss_ym, bce_loss_yt, valid_loss, time() - start_time),
                    file=self.log_file
                )
                print(
                    'epoch {:04d}/{:04d} | ym loss {:.4f} | yt loss {:.4f} | valid loss {:.4f} | time {:.4f} \n'
                    .format(epoch + 1, self.epochs, bce_loss_ym, bce_loss_yt, valid_loss, time() - start_time),
                )

            if (epoch + 1) % 2 == 0 or (epoch + 1) == self.epochs:
                torch.save(model.state_dict(), f'{self.save_name_pth}_epoch{epoch + 1}.pth')

        print('Total time cost for train and validation: {:.4f}'.format(time() - start_time), file=self.log_file)

    def valid_and_test(self, model, loader, df, confounder_info, pred_mode='do', beta=0.5, gamma=0.5):
        global total_loss
        if not self.is_train:
            loaded_file = self.save_name_pth + f'_epoch{self.load_epoch}.pth'
            model.load_state_dict(
                torch.load(loaded_file, map_location=self.device)
            )
            print('Loaded pth file:', str(loaded_file))
            print('Loaded pth file:', str(loaded_file), file=self.log_file)

        model.eval()

        eval_pred = []
        eval_true = []
        with (torch.no_grad()):
            start_time = time()
            if self.model_name == 'OURS':
                for idx, (x, diff, y) in enumerate(loader):
                    ym, yt, yt_mat = model.forward(x.to(self.device), diff.to(self.device))
                    y_pred = ym
                    if pred_mode == 'do':
                        out_new = beta * ym + (1 - beta) * nn.Sigmoid()(yt_mat).to(self.device)
                        out = torch.matmul(out_new, confounder_info.float().T.to(self.device)).to(
                            self.device)
                        out = out.unsqueeze(-1)
                        y_pred = out

                    elif pred_mode == 'w/o_ba':
                        out = beta * ym + (1 - beta) * yt
                        y_pred = out

                    y_pred = y_pred.detach().cpu()
                    loss = nn.BCELoss()(y_pred, y)

                    eval_pred.append(y_pred)
                    eval_true.append(y)

            elif self.model_name == 'TaFR':
                for idx, (x, diff, surv, y) in enumerate(loader):
                    ym = model.forward(x.to(self.device), diff.to(self.device))
                    if pred_mode == '!':  # 数据集不全
                        y_pred = ym
                    elif pred_mode == '~':
                        yt = surv[torch.arange(diff.shape[0]), diff].float().view(-1, 1)
                        y_pred = (gamma * ym.detach().cpu() + (1 - gamma) * yt.detach().cpu()).view(-1, 1)

                    y_pred = y_pred.detach().cpu()
                    loss = nn.BCELoss()(y_pred, y)
                    eval_pred.append(y_pred)
                    eval_true.append(y)

            eval_pred = torch.cat(eval_pred).squeeze()


            if (self.is_train > 0 and self.is_valid > 0) or self.is_train == 0:
                df['pred'] = eval_pred.tolist()

                if self.dataset == 'kuairand_pure':
                    TopK(5, df, self.log_file).evaluate()
                    TopK(10, df, self.log_file).evaluate()

                elif self.dataset == 'kuairand_1k':
                    TopK(300, df, self.log_file).evaluate()
                    TopK(500, df, self.log_file).evaluate()

        return loss

    def complexity_analysis(self, model, loader):
        global batch_size, size_x
        for batch_idx, (x, diff, targets) in enumerate(loader):
            flops = FlopCountAnalysis(model, (x.to(self.device), diff.to(self.device)))
            print(f"FLOPs: {flops.total()}")
            print(f"FLOPs: {flops.total()}", file=self.log_file)
            break

        model.eval()
        print('Params:', parameter_count_table(model))
        print('Params:', parameter_count_table(model), file=self.log_file)
