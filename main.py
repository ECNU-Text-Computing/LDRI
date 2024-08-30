import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from preprocessing.loader import Loader
from preprocessing.generator import Generator
from utils.train import *
from utils.seed import set_seed
from config import args
from models.LDRI import LDRI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # set seed
    set_seed(args.seed)

    print('Device:', device)

    data = Generator(args.dataset, use_dense=False)
    train_loader, valid_loader, test_loader, train_df, valid_df, test_df, confounder_info \
        = data.wrapper(batch_size=args.batch_size, num_samples=args.num_samples)
    feature_size_map = getattr(data, 'feature_size_map')
    feature_size_map_item = getattr(data, 'feature_size_map_item')
    n_day = 30

    model = LDRI(device=device,
                 feature_size_map=feature_size_map,
                 feature_size_map_item=feature_size_map_item,
                 n_day=n_day, n=args.n,
                 embed_dim_sparse=16,
                 use_dense=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # train and test model
    TrainAndTest(model=model, backbone=args.backbone, device=device,
                 optimizer=optimizer, criterion=criterion, seed=args.seed,
                 train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, train_df=train_df, valid_df=valid_df, test_df=test_df,
                 dataset=args.dataset, num_samples=args.num_samples, epochs=args.epochs,
                 is_train=args.is_train, is_valid=args.is_valid, load_epoch=args.load_epoch,
                 confounder_info=confounder_info,
                 n=int(args.n), alpha=args.alpha, beta=args.beta
                 )
