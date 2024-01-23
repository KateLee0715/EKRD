import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="EKRD")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="amazon-book", help="Choose a dataset:[movie-lens,last-fm,amazon-book]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
    parser.add_argument('--reg', default=1e-8, type=float, help='weight decay regularizer')
    parser.add_argument('--cdreg', default=1e-3, type=float, help='contrastive distillation reg weight')
    parser.add_argument('--lsreg', default=1e-3, type=float, help='contrastive distillation reg weight')
    parser.add_argument('--aglreg', default=1e-3, type=float, help='contrastive distillation reg weight')
    parser.add_argument('--softreg', default=1e-2, type=float, help='soft-target-based distillation reg weight')
    parser.add_argument('--decay', default=1.0, type=float, help='regularization per-epoch decay')
    parser.add_argument('--latdim', default=64, type=int, help='embedding size')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--teacher_model', default='KGIN', help='model name for teacher to load')
    parser.add_argument('--topk', default=20, type=int, help='K of top K')
    parser.add_argument('--topRange', default=100000, type=int, help='adaptive pick range')
    parser.add_argument('--tempsoft', default=0.03, type=float,
                        help='temperature for soft binary classification in distillation')
    parser.add_argument('--tempcd', default=0.1, type=float, help='temperature for contrastive distillation')
    parser.add_argument('--templs', default=0.1, type=float, help='temperature for contrastive distillation')
    parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--inverse_r", type=int, default=1, help="consider inverse relation or not")
    parser.add_argument("--entity_num_per_item", type=int, default=10, help="entity num per item")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")
    parser.add_argument('--laplacian_type', type=str, default='random-walk',
                        help='Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.')
    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--aggregation_type', type=str, default='bi-interaction',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--kgat_mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")

    return parser.parse_args()


args = parse_args()
