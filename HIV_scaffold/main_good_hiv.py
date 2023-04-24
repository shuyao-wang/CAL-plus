# from datasets import get_dataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
import opts
import warnings
import numpy as np
import random

warnings.filterwarnings('ignore')
import time
import sys

sys.path.append("..")
from GOOD.data.good_datasets.good_hiv import GOODHIV
from torch.optim import Adam
import torch
import torch.nn.functional as F
from cal_gnn import HivCausalGIN
from torch.optim.lr_scheduler import CosineAnnealingLR
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def eval(model, evaluator, loader, device):

    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model.eval_forward(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    output = evaluator.eval(input_dict)
    return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss().to(device)


def main(args):

    prototype = None
    memory_bank = 0.5 * torch.ones(args.batch_size * args.me_batch_n,
                                   args.hidden).to(device)
    path = "dataset"
    dataset, meta_info = GOODHIV.load(path,
                                      domain=args.domain,
                                      shift=args.shift,
                                      generate=False)
    num_class = 1
    eval_metric = "rocauc"
    eval_name = "ogbg-molhiv"
    evaluator = Evaluator('ogbg-molhiv')

    train_loader = DataLoader(dataset["train"],
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader_ood = DataLoader(dataset["val"],
                                  batch_size=args.batch_size,
                                  shuffle=False)

    test_loader_ood = DataLoader(dataset["test"],
                                 batch_size=args.batch_size,
                                 shuffle=False)

    # pdb.set_trace()
    model = HivCausalGIN(hidden_in=args.hidden,
                         hidden_out=num_class,
                         hidden=args.hidden,
                         num_layer=args.layers,
                         cls_layer=args.cls_layer).to(device)
    results = {
        'highest_valid_ood': 0,
        'update_test_ood': 0,
        'update_epoch_ood': 0,
    }

    BCELoss = torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer,
                                     T_max=args.epochs,
                                     eta_min=args.min_lr,
                                     last_epoch=-1,
                                     verbose=False)
    for epoch in range(1, args.epochs + 1):

        model.train()
        total_loss = 0
        total_p = 0
        # show = int(float(len(train_loader)) / 4.0)
        for step, data in enumerate(train_loader):

            optimizer.zero_grad()
            data = data.to(device)

            if args.memory == True:
                start = time.time()
                # k = 0
                for step_, data_ in enumerate(train_loader):

                    # if step_ in xc_id:
                    if step_ < args.me_batch_n:
                        data_ = data_.to(device)
                        if data_.x.shape[0] == 1:
                            pass
                        else:
                            with torch.no_grad():

                                memory_bank[step_ * args.batch_size:(
                                    step_ * args.batch_size +
                                    data_.y.view(-1).shape[0]
                                )] = model.forward_xc(data_)
                                # k += 1

                    else:
                        break
                time_ = time.time() - start
                num = args.batch_size * args.me_batch_n
                l = [i for i in range(num)]
                random.shuffle(l)
                random_idx = torch.tensor(l)
                xc = memory_bank[random_idx[0:data.y.view(-1).shape[0]]]
            else:
                xc = model.forward_xc(data)

            xo = model.forward_xo(data)
            c_logs, o_logs, co_logs = model(xo, xc)

            if args.prototype == True:
                class_causal = class_split(data.y, xo, 2)
                p_loss = loss_prototype(prototype, class_causal, criterion,
                                        args)
                p_loss.backward(retain_graph=True)

            else:
                p_loss = 0

            uniform_target = torch.ones_like(
                c_logs, dtype=torch.float).to(device) / model.num_classes
            # pdb.set_trace()
            c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
            is_labeled = data.y == data.y
            o_loss = BCELoss(
                o_logs.to(torch.float32)[is_labeled],
                data.y.to(torch.float32)[is_labeled])
            co_loss = BCELoss(
                co_logs.to(torch.float32)[is_labeled],
                data.y.to(torch.float32)[is_labeled])
            # o_loss = F.nll_loss(o_logs, one_hot_target)
            # co_loss = F.nll_loss(co_logs, one_hot_target)
            loss = args.c * c_loss + args.o * o_loss + args.co * co_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_p += p_loss

            # if step % show == 0:
            #     print(
            #         "Epoch:[{}/{}] Train Iter:[{:<3}/{}] Loss:[{:.4f}]".format(
            #             epoch, args.epochs, step, len(train_loader),
            #             total_loss / (step + 1)))
        # total_loss = total_loss / len(train_loader.dataset)
        # total_p = total_p / len(train_loader.dataset)
        valid_result_ood = eval(model, evaluator, valid_loader_ood,
                                device)[eval_metric]

        test_result_ood = eval(model, evaluator, test_loader_ood,
                               device)[eval_metric]
        lr_scheduler.step()

        if valid_result_ood > results[
                'highest_valid_ood'] and epoch > args.pretrain:
            results['highest_valid_ood'] = valid_result_ood
            results['update_test_ood'] = test_result_ood
            results['update_epoch_ood'] = epoch

        print("-" * 150)
        print(
            "Epoch:[{}/{}], lr:[{:.6f}] Loss:[{:.4f}] P_Loss:[{:.4f}] valid:[{:.2f}], test:[{:.2f}] | Best val:[{:.2f}] Update test:[{:.2f}] at:[{}]"
            .format(epoch, args.epochs, optimizer.param_groups[0]['lr'],
                    total_loss, total_p, valid_result_ood * 100,
                    test_result_ood * 100, results['highest_valid_ood'] * 100,
                    results['update_test_ood'] * 100,
                    results['update_epoch_ood']))
        print("-" * 150)

    print("mwy: Update test:[{:.2f}] at epoch:[{}]".format(
        results['update_test_ood'] * 100, results['update_epoch_ood']))
    return results['update_test_ood']


def class_split(y, causal_feature, num_classes):
    class_causal = {}
    j = 0
    for i in range(num_classes):
        k = np.where(y.view(-1).cpu() == i)
        if len(k[0]) == 0:
            continue
        else:
            class_idx = torch.tensor(k).view(-1)
            calss_causal_feature = causal_feature[class_idx]
            class_causal[j] = calss_causal_feature
            j += 1
    return class_causal


def softmax_with_temperature(input, t=1, axis=-1):
    ex = torch.exp(input / t)
    sum = torch.sum(ex, axis=axis)
    return ex / sum


#update prototypes
def loss_prototype(prototype, class_causal, criterion, args):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    distance = None
    if prototype is None:
        prototype = [
            torch.mean(class_causal[key].to(torch.float32),
                       dim=0,
                       keepdim=True).detach() for key in class_causal
        ]
    else:
        for i in range(len(class_causal)):
            cosine = cos(prototype[i], class_causal[i])
            weights_proto = softmax_with_temperature(cosine,
                                                     t=5).reshape(1, -1)
            prototype[i] = torch.mm(weights_proto, class_causal[i]).detach()
    #prototype_ = torch.cat(prototype, dim=0).softmax(dim=-1)
    #kl_div_loss
    prototype_ = torch.cat(prototype, dim=0)
    for i in range(len(class_causal)):

        prototype_ = torch.cat(
            (prototype_[i:i + 1], prototype_[0:i], prototype_[i + 1:]), 0)
        distance_ = torch.einsum('nc,kc->nk', [
            nn.functional.normalize(class_causal[i], dim=1),
            nn.functional.normalize(prototype_, dim=1)
        ])
        #distance_ /= 5
        if distance is None:
            distance = F.softmax(distance_, dim=1)
        else:
            distance = torch.cat((distance, F.softmax(distance_, dim=1)), 0)
    labels = torch.zeros(distance.shape[0], dtype=torch.long).cuda()
    if len(class_causal) == 1:
        loss = torch.mean(distance, dim=0)[0]

    else:
        loss = criterion(distance, labels)
    return args.beta * loss * 0.2


def config_and_run(args):

    final_test_acc_ood = []
    for _ in range(args.trails):
        test_auc_ood = main(args)

        final_test_acc_ood.append(test_auc_ood)
    print("mwy finall: Test ACC OOD: [{:.2f}±{:.2f}]".format(
        np.mean(final_test_acc_ood) * 100,
        np.std(final_test_acc_ood) * 100))
    print(" all OOD:{}".format(final_test_acc_ood))


if __name__ == '__main__':
    args = opts.parse_args()
    config_and_run(args)
    print(
        "settings | beta:[{}]  n:[{}]  prototype/memory:[{}/{}]   batch_size:[{}]  hidden:[{}] lr:[{}] min_lr:[{}] weight_decay[{}] "
        .format(str(args.beta), str(args.me_batch_n), str(args.prototype),
                str(args.memory), str(args.batch_size), str(args.hidden),
                str(args.lr), str(args.min_lr), str(args.weight_decay)))
