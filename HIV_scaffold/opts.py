import argparse
import numpy as np
import random
import torch
from itertools import product


def parse_args():

    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    parser.add_argument('--shift',
                        type=str,
                        default="concept",
                        help="irm, dro, base")
    parser.add_argument('--domain',
                        type=str,
                        default="scaffold",
                        help="irm, dro, base")
    parser.add_argument('--dataset', type=str, default="hiv")
    parser.add_argument('--prototype', type=str2bool, default="True")
    parser.add_argument('--memory', type=str2bool, default="True")
    parser.add_argument('--me_batch_n', type=int, default=1)

    parser.add_argument('--trails', type=int, default=1)
    parser.add_argument('--pretrain', type=int, default=20)
    # parser.add_argument('--step_size', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    # #################### toy example #######################
    parser.add_argument('--cls_layer', type=int, default=2)
    parser.add_argument('--beta', type=float, default=1)
    # parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)

    # parser.add_argument('--the', type=int, default=0)
    # parser.add_argument('--with_random', type=str2bool, default=True)
    # parser.add_argument('--eval_random', type=str2bool, default=False)
    # parser.add_argument('--normalize', type=str2bool, default=False)
    # parser.add_argument('--save_model', type=str2bool, default=False)
    # parser.add_argument('--inference', type=str2bool, default=False)
    # parser.add_argument('--without_node_attention',
    #                     type=str2bool,
    #                     default=False)
    # parser.add_argument('--without_edge_attention',
    #                     type=str2bool,
    #                     default=False)

    # parser.add_argument('--k', type=int, default=3)
    #################### Causal GNN settings #######################
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--c', type=float, default=0.5)
    parser.add_argument('--o', type=float, default=1.0)
    parser.add_argument('--co', type=float, default=0.5)
    parser.add_argument('--harf_hidden', type=float, default=0.5)
    parser.add_argument('--cat_or_add', type=str, default="add")
    ##################### baseline training ######################
    # parser.add_argument('--num_layers', type=int, default=3)

    # parser.add_argument('--folds', type=int, default=10)
    # parser.add_argument('--fc_num', type=str, default="222")
    # parser.add_argument('--data_root', type=str, default="data")
    # parser.add_argument('--save_dir', type=str, default="debug")

    # parser.add_argument('--epoch_select', type=str, default='test_max')
    # parser.add_argument('--model', type=str, default="GCN", help="GCN, GIN")
    parser.add_argument('--hidden', type=int, default=300)

    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--global_pool', type=str, default="sum")
    args = parser.parse_args()
    print_args(args)
    setup_seed(args.seed)
    return args


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
