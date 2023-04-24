import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
import warnings

warnings.filterwarnings('ignore')
from torch.optim.lr_scheduler import CosineAnnealingLR
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch import tensor
from train_epoch import train_causal_epoch
import numpy as np
import os
from opts_GOOD import print_args, parse_args

from model_GOOD import Causal
import time
import random
import pdb
from utils import *
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def _init_fn(worker_id):
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data,
                                              a=0,
                                              mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss().to(device)

# def eval(model, evaluator, loader, device):
#     model.eval()

#     y_true = []
#     y_pred = []
#     for step, batch in enumerate(loader):
#         batch = batch.to(device)
#         if batch.x.shape[0] == 1:
#             pass
#         else:
#             with torch.no_grad():
#                 pred = model.forward_causal(batch)
#             y_true.append(batch.y.view(pred.shape).detach().cpu())
#             y_pred.append(pred.detach().cpu())
#     y_true = torch.cat(y_true, dim=0).numpy()
#     y_pred = torch.cat(y_pred, dim=0).numpy()
#     input_dict = {"y_true": y_true, "y_pred": y_pred}
#     output = evaluator.eval(input_dict)
#     return output


def main(args, trail):

    set_seed(args.seed)
    dataset = PygGraphPropPredDataset(name="ogbg-molbbbp", root='dataset')
    args.num_classes = 2
    args.eval_metric = "rocauc"
    args.eval_name = "ogbg-molbbbp"
    args.dim_node = 9
    if args.domain == "scaffold":
        split_idx = dataset.get_idx_split()
    else:
        split_idx = size_split_idx(dataset, "ls")

    random_guess = 1.0 / args.num_classes
    evaluator = Evaluator(args.eval_name)
    train_loader = DataLoader(dataset[split_idx["train"]],
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              worker_init_fn=_init_fn)
    valid_loader_ood = DataLoader(dataset[split_idx["valid"]],
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  worker_init_fn=_init_fn)
    test_loader_ood = DataLoader(dataset[split_idx["test"]],
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 worker_init_fn=_init_fn)

    model = Causal(args.dim_node, args.num_classes, args).to(device)
    init_weights(model, 'kaiming', init_gain=0.02)
    optimizer = Adam(model.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer,
                                     T_max=args.epochs,
                                     eta_min=args.min_lr,
                                     last_epoch=-1,
                                     verbose=False)

    results = {'highest_valid': 0, 'update_test': 0, 'update_epoch': 0}
    start_time = time.time()
    prototype = None
    memory_bank = 0.5 * torch.ones(args.batch_size * args.me_batch_n,
                                   args.hidden).to(device)
    for epoch in range(1, args.epochs + 1):
        start_time_local = time.time()
        train_loss, loss_c, loss_o, loss_co, loss_p = train_causal_epoch(
            model, optimizer, train_loader, device, memory_bank, prototype,
            criterion, args)

        valid_result = eval(model, valid_loader_ood, device, args, evaluator)
        test_result = eval(model, test_loader_ood, device, args, evaluator)
        lr_scheduler.step()
        if epoch > args.pretrain:
            if valid_result > results['highest_valid']:
                results['highest_valid'] = valid_result
                results['update_test'] = test_result
                results['update_epoch'] = epoch
        else:
            pass

        print("-" * 150)
        print(
            "Causal | dataset:[{}] fold:[{}] | Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}+{:.4f}] Valid:[{:.4f}] Test:[{:.4f}]  (RG:{:.4f}) | Best Valid:[{:.4f}] Update test:[{:.4f}] at Epoch:[{}] | epoch time[{:.2f}min "
            .format(args.dataset, trail, epoch, args.epochs, train_loss,
                    loss_c, loss_o, loss_co, loss_p, valid_result, test_result,
                    random_guess, results['highest_valid'],
                    results['update_test'], results['update_epoch'],
                    (time.time() - start_time_local) / 60))

        print("-" * 150)
    total_time = time.time() - start_time

    print(
        "mwy: Causal fold:[{}] | Dataset:[{}] | Update Test:[{:.4f}] at epoch [{}] | (RG:{:.4f}) | Total time:{}"
        .format(trail, args.dataset, results['update_test'],
                results['update_epoch'], random_guess,
                time.strftime('%H:%M:%S', time.gmtime(total_time))))

    print("-" * 150)
    print('\n')
    # final_test_iid.append(results['update_test_iid'])
    return results['update_test']


def config_and_run(args):

    print_args(args)
    # set_seed(args.seed)
    final_test = []
    for trail in range(args.trails):
        test_result = main(args, trail + 1)
        final_test.append(test_result)
    print("mwy finall: Test result: [{:.2f}±{:.2f}]".format(
        np.mean(final_test) * 100,
        np.std(final_test) * 100))
    print("ALL OOD:{}".format(final_test))


def eval(model, loader, device, args, evaluator):
    model.eval()
    # eval_random = args.eval_random
    if args.eval_metric == "acc":
        correct_o = 0
        for data in loader:
            data = data.to(device)
            if data.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    o_logs = model.eval_forward(data)

                    pred_o = o_logs.max(1)[1]

                    correct_o += pred_o.eq(data.y.view(-1)).sum().item()

        output_o = correct_o / len(loader.dataset)

    elif args.eval_metric == "rocauc":
        y_true, o_pred = [], []

        for data in loader:
            data = data.to(device)
            if data.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    o_logs = model.eval_forward(data)

                    o_logs = torch.max(o_logs, 1, keepdim=True)[0]

                    y_true.append(data.y.view(o_logs.shape).detach().cpu())

                    o_pred.append(o_logs.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        o_pred = torch.cat(o_pred, dim=0).numpy()
        input_dict_o = {"y_true": y_true, "y_pred": o_pred}
        # try:
        output_o = evaluator.eval(input_dict_o)[args.eval_metric]
        # except:
        #     pdb.set_trace()
    else:
        assert False

    return output_o


if __name__ == "__main__":
    args = parse_args()
    config_and_run(args)
    print(
        "settings | beta:[{}]  n:[{}]  prototype/memory:[{}/{}]  dim_node:[{}] num_classes:[{}] batch_size:[{}]  hidden:[{}] lr:[{}] min_lr:[{}] weight_decay[{}] pretrain[{}] co[{}] c[{}] o[{}]"
        .format(str(args.beta), str(args.me_batch_n), str(args.prototype),
                str(args.memory), str(args.dim_node), str(args.num_classes),
                str(args.batch_size), str(args.hidden), str(args.lr),
                str(args.min_lr), str(args.weight_decay), str(args.pretrain),
                str(args.co), str(args.c), str(args.o)))

    print("-" * 150)
