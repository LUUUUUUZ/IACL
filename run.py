from srs.utils.argparse import ArgumentParser
from pathlib import Path
import torch
import numpy as np
import random
import time
# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0)

parser = ArgumentParser()
parser.add_argument('--model', required=True, help='the prediction model')
parser.add_argument(
    '--num-cluster', type=int, default=1024, help='number of cluster'
)
parser.add_argument('--augment-type', default='substitute', type=str, help="chosen from: mask, crop, reorder, substitute, insert")
parser.add_argument('--similarity-method', default='ItemCF', type=str, help='chosen from: ItemTrans, ItemCF, ItemCF_IUF')
parser.add_argument('--hybrid-epoch', default=0, type=int, help='offline:0, online:1, hybrid epoch')
parser.add_argument(
    '--cluster-interval', type=int, default=3, help='number of cluster'
)
parser.add_argument(
    '--alpha', type=float, default=0.1, help='sequance contrastive loss ratio'
)
parser.add_argument(
    '--beta', type=float, default=0.1, help='intent contrastive loss ratio'
)
parser.add_argument(
    '--noise-ratio', type=float, default=0.0, help='noise ratio during test'
)
parser.add_argument(
    '--dataset-dir', type=Path, required=True, help='the dataset set directory'
)
parser.add_argument(
    '--embedding-dim', type=int, default=128, help='the dimensionality of embeddings'
)
parser.add_argument(
    '--feat-drop', type=float, default=0.5, help='the dropout ratio for input features'
)
parser.add_argument(
    '--num-layers',
    type=int,
    default=1,
    help='the number of HGNN layers in the KGE component',
)
parser.add_argument(
    '--num-fmlp-layers',
    type=int,
    default=1,
    help='the number of fmlp layers',
)
parser.add_argument(
    '--num-neighbors',
    default='10',
    help='the number of neighbors to sample at each layer.'
    ' Give an integer if the number is the same for all layers.'
    ' Give a list of integers separated by commas if this number is different at different layers, e.g., 10,10,5'
)
parser.add_argument(
    '--model-args',
    type=str,
    default='{}',
    help="the extra arguments passed to the model's initializer."
    ' Will be evaluated as a dictionary.',
)
parser.add_argument('--batch-size', type=int, default=128, help='the batch size')
parser.add_argument(
    '--epochs', type=int, default=30, help='the maximum number of training epochs'
)
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
parser.add_argument(
    '--weight-decay',
    type=float,
    default=1e-4,
    help='the weight decay for the optimizer',
)
parser.add_argument(
    '--patience',
    type=int,
    default=3,
    help='stop training if the performance does not improve in this number of consecutive epochs',
)
parser.add_argument(
    '--Ks',
    default='5,10,20',
    help='the values of K in evaluation metrics, separated by commas'
)
parser.add_argument(
    '--ignore-list',
    default='bias,batch_norm,activation',
    help='the names of parameters excluded from being regularized',
)
parser.add_argument(
    '--log-level',
    choices=['debug', 'info', 'warning', 'error'],
    default='debug',
    help='the log level',
)
parser.add_argument(
    '--log-interval',
    type=int,
    default=1000,
    help='if log level is info or debug, print training information after every this number of iterations',
)
parser.add_argument(
    '--device', type=int, default=0, help='the index of GPU device (-1 for CPU)'
)
parser.add_argument(
    '--num-workers',
    type=int,
    default=1,
    help='the number of processes for data loaders',
)
parser.add_argument(
    '--OTF',
    action='store_true',
    help='compute KG embeddings on the fly instead of precomputing them before inference to save memory',
)
parser.add_argument(
    '--model_drop', type=float, default=0.5, help='the dropout ratio for model'
)
args = parser.parse_args()
args.model_args = eval(args.model_args)
args.num_neighbors = [int(x) for x in args.num_neighbors.split(',')]
args.Ks = [int(K) for K in args.Ks.split(',')]
args.ignore_list = [x.strip() for x in args.ignore_list.split(',') if x.strip() != '']

import logging
import importlib
import os

module = importlib.import_module(f'srs.models.{args.model}')
config = module.config
for k, v in vars(args).items():
    config[k] = v
args = config

log_level = getattr(logging, args.log_level.upper(), None)
logging.basicConfig(format='%(message)s', level=log_level)
logging.debug(args)

loca=time.time
loca=time.strftime('%Y-%m-%d-%H-%M-%S')
dataset_name = str(args.dataset_dir).split('/')[-1]
args.log_file = os.path.join( './outputs/', str(loca)+ '-'+ dataset_name+'-'+str(args.model)+ ".txt")
with open(args.log_file,'a') as f:
    f.write(str(args)+'\n')

import torch as th
from torch.utils.data import DataLoader
from srs.layers.seframe import SEFrame
from srs.utils.data.load import read_dataset, AugmentedDataset, AnonymousAugmentedDataset
from srs.utils.data.augmentation import OfflineItemSimilarity, OnlineItemSimilarity
from srs.utils.train_runner_cl import TrainRunnerCL

args.gpu_id = args.device # int 
args.device = (
    th.device('cpu') if args.device < 0 else th.device(f'cuda:{args.device}')
)
args.prepare_batch = args.prepare_batch_factory(args.device)

logging.info(f'reading dataset {args.dataset_dir}...')
df_train, df_valid, df_test, stats = read_dataset(args.dataset_dir)

if issubclass(args.Model, SEFrame):
    from srs.utils.data.load import (read_social_network, build_knowledge_graph)

    social_network = read_social_network(args.dataset_dir / 'edges.txt')
    args.knowledge_graph = build_knowledge_graph(df_train, social_network)
    # import pdb;pdb.set_trace()

elif args.Model.__name__ == 'DGRec':
    from srs.utils.data.load import (
        compute_visible_time_list_and_in_neighbors,
        filter_invalid_sessions,
    )

    visible_time_list, in_neighbors = compute_visible_time_list_and_in_neighbors(
        df_train, args.dataset_dir, args.num_layers
    )
    args.visible_time_list = visible_time_list
    args.in_neighbors = in_neighbors
    args.uid2sessions = [{
        'sids': df['sessionId'].values,
        'sessions': df['items'].values
    } for _, df in df_train.groupby('userId')]
    L_hop_visible_time = visible_time_list[args.num_layers]

    df_train, df_valid, df_test = filter_invalid_sessions(
        df_train, df_valid, df_test, L_hop_visible_time=L_hop_visible_time
    )

args.num_users = getattr(stats, 'num_users', None)
args.num_items = stats.num_items
args.max_len = stats.max_len

model = args.Model(**args, **args.model_args)
model = model.to(args.device)
logging.debug(model)

if args.num_users is None:
    train_set = AnonymousAugmentedDataset(df_train)
    cluster_set = AnonymousAugmentedDataset(df_train) 
    valid_set = AnonymousAugmentedDataset(df_valid)
    test_set = AnonymousAugmentedDataset(df_test)
else:
    read_sid = args.Model.__name__ == 'DGRec'
    train_set = AugmentedDataset(df_train, read_sid)
    cluster_set = AugmentedDataset(df_train, read_sid) 
    valid_set = AugmentedDataset(df_valid, read_sid)
    test_set = AugmentedDataset(df_test, read_sid)

args.model = model

logging.debug('using batch sampler')
batch_sampler = config.BatchSampler(
    train_set, batch_size=args.batch_size, drop_last=True, seed=0
)

# offline:
# -----------   pre-computation for item similarity   ------------ #
args.similarity_path = os.path.join(dataset_name+'_'+args.similarity_method+'_similarity.pkl')

offline_similarity_model = OfflineItemSimilarity(df=df_train, **args)
args.offline_similarity_model = offline_similarity_model

# # -----------   online based on shared item embedding for item similarity --------- #
online_similarity_model = OnlineItemSimilarity(item_size=args.num_items, **args)
args.online_similarity_model = online_similarity_model


if args.hybrid_epoch == 0:
    collate_fn = args.CollateFn(similarity_model=args.offline_similarity_model, **args)
    collate_train = collate_fn.collate_train
    collate_test = collate_fn.collate_test

    train_loader = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        collate_fn=collate_train,
        num_workers=args.num_workers,
    )
    cluster_loader = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        collate_fn=collate_train,
        num_workers=args.num_workers,
    )

elif args.hybrid_epoch == 1:
   
    collate_fn = args.CollateFn(similarity_model=args.online_similarity_model, **args)
    collate_train = collate_fn.collate_train
    collate_test = collate_fn.collate_test

    train_loader = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        collate_fn=collate_train,
        num_workers=args.num_workers,
    )
    cluster_loader = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        collate_fn=collate_train,
        num_workers=args.num_workers,
    )

else:
    collate_fn1 = args.CollateFn(similarity_model=args.offline_similarity_model, **args)
    collate_train1 = collate_fn1.collate_train
    
    collate_test = collate_fn1.collate_test

    train_loader1 = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        collate_fn=collate_train1,
        num_workers=args.num_workers,
    )
    cluster_loader1 = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        collate_fn=collate_train1,
        num_workers=args.num_workers,
    ) 

    collate_fn2 = args.CollateFn(similarity_model=args.online_similarity_model, **args)
    collate_train2 = collate_fn2.collate_train

    train_loader2 = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        collate_fn=collate_train2,
        num_workers=args.num_workers,
    )
    cluster_loader2 = DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        collate_fn=collate_train2,
        num_workers=args.num_workers,
    ) 

    train_loader = [train_loader1, train_loader2]
    cluster_loader = [cluster_loader1, cluster_loader2]

valid_loader = DataLoader(
    valid_set,
    batch_size=args.batch_size,
    collate_fn=collate_test,
    num_workers=args.num_workers,
    drop_last=False,
    shuffle=False,
)

test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    collate_fn=collate_test,
    num_workers=args.num_workers,
    drop_last=False,
    shuffle=False,
)

runner = TrainRunnerCL(train_loader, valid_loader, test_loader, cluster_loader, **args)
logging.info('start training')
results = runner.train(args.epochs, log_interval=args.log_interval)
