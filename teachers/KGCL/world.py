import os
from os.path import join
import torch
from parse import args

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ROOT_PATH = "./"
DATA_PATH = join(ROOT_PATH, 'data')
FILE_PATH = join(ROOT_PATH, 'checkpoints')

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

config = {}
config['latent_dim_rec'] = args.latdim
config['lightGCN_n_layers']= args.gnn_layer
# config['dropout'] = args.dropout
# config['keep_prob']  = args.keepprob
# config['A_n_fold'] = args.a_fold
# config['test_u_batch_size'] = args.testbatch
# config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = 0
config['A_split'] = False

kgcn = "RGAT"
train_trans = True
entity_num_per_item = 10
args.entity_num_per_item = entity_num_per_item
# WEIGHTED (-MIX) \ RANDOM \ ITEM-BI \ PGRACE \NO
uicontrast = "RANDOM"
kgc_enable = True
kgc_joint = True
kgc_temp = 0.2
use_kgc_pretrain = False
pretrain_kgc = False
kg_p_drop = 0.5
ui_p_drop = 0.001
ssl_reg = 0.1

dataset = args.dataset
if dataset == 'MIND':
    # config['lr'] = 5e-4
    # config['decay'] = 1e-3

    uicontrast = "WEIGHTED-MIX"
    kgc_enable = True
    kgc_joint = True
    use_kgc_pretrain = False
    entity_num_per_item = 6
    args.entity_num_per_item = entity_num_per_item
    # [0.06, 0.08, 0.1]
    ssl_reg = 0.06
    kgc_temp = 0.2
    # [0.3, 0.5, 0.7]
    kg_p_drop = 0.5
    # [0.1, 0.2, 0.4]
    ui_p_drop = 0.4
    mix_ratio = 1 - ui_p_drop - 0
    test_start_epoch = 1
    early_stop_cnt = 3

elif dataset == 'amazon-book':
    uicontrast = "WEIGHTED"
    ui_p_drop = 0.05
    mix_ratio = 0.75
    test_start_epoch = 15
    early_stop_cnt = 5

elif dataset == 'yelp2018':
    uicontrast = "WEIGHTED"
    ui_p_drop = 0.1
    test_start_epoch = 25
    early_stop_cnt = 5