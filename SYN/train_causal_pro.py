import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import tensor
import numpy as np
from data_utils.utils import k_fold, num_graphs
import random
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_causal_syn(train_set, val_set, test_set, model_func=None, args=None):

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
        
    model = model_func(args.feature_dim, args.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=-1, verbose=False)

    # start_time = time.time()
    prototype = None
    memory_bank = 0.5 * torch.ones(args.batch_size * args.me_batch_n,
                                   args.hidden).to(device)

    best_val_acc, update_test_acc_o, update_epoch = 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        
        train_loss, loss_c, loss_o, loss_co, loss_p = train_causal_epoch(model, optimizer, train_loader, device, memory_bank, prototype, args)

        val_acc_o = eval_acc_causal(model, val_loader, device, args)
        test_acc_o = eval_acc_causal(model, test_loader, device, args)
        # pdb.set_trace()
        lr_scheduler.step()
        if val_acc_o > best_val_acc:
            best_val_acc = val_acc_o
            # update_test_acc_co = test_acc_co
            # update_test_acc_c = test_acc_c
            update_test_acc_o = test_acc_o
            update_epoch = epoch
            
        print("BIAS:[{:.2f}] | Model:[{}] Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}+{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Update Test:[o:{:.2f}] at Epoch:[{}] | lr:{:.6f}"
                .format(args.bias,
                        args.model,
                        epoch, 
                        args.epochs,
                        train_loss,
                        loss_c,
                        loss_o,
                        loss_co,
                        loss_p,
                        val_acc_o * 100,
                        test_acc_o * 100,  
                        update_test_acc_o * 100, 
                        update_epoch,
                        optimizer.param_groups[0]['lr']))

    print("syd: BIAS:[{:.2f}] | Val acc:[{:.2f}] Test acc:[o:{:.2f}] at epoch:[{}]"
        .format(args.bias,
                val_acc_o * 100,
                update_test_acc_o * 100, 
                update_epoch))


def train_causal_epoch(model, optimizer, loader, device, memory_bank, prototype, args):
    
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    total_loss_p = 0
    # correct_o = 0
    
    for it, data in enumerate(loader):
        
        optimizer.zero_grad()
        data = data.to(device)

        if data.feat.shape[0] == 1:
            pass
        else:
            if args.memory == True:
                # start = time.time()
                # k = 0
                for step_, data_ in enumerate(loader):

                    # if step_ in xc_id:
                    if step_ < args.me_batch_n:
                        data_ = data_.to(device)
                        if data_.feat.shape[0] == 1:
                            pass
                        else:
                            with torch.no_grad():
                                # pdb.set_trace()
                                memory_bank[step_ * args.batch_size:(
                                    step_ * args.batch_size +
                                    data_.y.view(-1).shape[0]
                                )] = model.forward_xc(data_)
                                # k += 1

                    else:
                        break
                # time_ = time.time() - start
                num = args.batch_size * args.me_batch_n
                l = [i for i in range(num)]
                random.shuffle(l)
                random_idx = torch.tensor(l)
                xc = memory_bank[random_idx[0:data.y.view(-1).shape[0]]]

            else:
                xc = model.forward_xc(data)

            xo = model.forward_xo(data)

            if args.prototype == True:
                # pdb.set_trace()
                class_causal = class_split(data.y, xo, args.num_classes)
                p_loss = loss_prototype(prototype, class_causal, args)
                # pdb.set_trace()
                p_loss.backward(retain_graph=True)
            else:
                p_loss = 0




            one_hot_target = data.y.view(-1)
            c_logs, o_logs, co_logs = model.forward(xo, xc)
            uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
            target_rep = data.y.to(torch.float).repeat_interleave(data.batch[-1] + 1, dim=0).view(-1).type(torch.int64)

            c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
            o_loss = F.nll_loss(o_logs, one_hot_target)
            co_loss = F.nll_loss(co_logs, target_rep)
            loss = args.c * c_loss + args.o * o_loss + args.co * co_loss

            # pred_o = o_logs.max(1)[1]
            # correct_o += pred_o.eq(data.y.view(-1)).sum().item()
            loss.backward()
            total_loss += loss.item() * num_graphs(data)
            total_loss_c += c_loss.item() * num_graphs(data)
            total_loss_o += o_loss.item() * num_graphs(data)
            total_loss_co += co_loss.item() * num_graphs(data)
            total_loss_p += p_loss.item() * num_graphs(data)
            optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    total_loss_p = total_loss_p / num
    # correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, total_loss_p

def eval_acc_causal(model, loader, device, args):
    
    model.eval()
    correct_o = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():

            o_logs = model.eval_forward(data)
            # pred = co_logs.max(1)[1]
            # pred_c = c_logs.max(1)[1] 
            pred_o = o_logs.max(1)[1] 
        # correct += pred.eq(data.y.view(-1)).sum().item()
        # correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()

    # acc_co = correct / len(loader.dataset)
    # acc_c = correct_c / len(loader.dataset)
    acc_o = correct_o / len(loader.dataset)

    return acc_o


def class_split(y, causal_feature, num_classes):
    # pdb.set_trace()
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
    # pdb.set_trace()
    return class_causal


def softmax_with_temperature(input, t=1, axis=-1):
    ex = torch.exp(input / t)
    sum = torch.sum(ex, axis=axis)
    return ex / sum


#update prototypes
def loss_prototype(prototype, class_causal, args):
    # pdb.set_trace()
    criterion = nn.CrossEntropyLoss().to(device)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    distance = None
    pos_loss = 0

    neg_loss = 0
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
        #     x_ = F.log_softmax(class_causal[i], dim=-1)
        #     target_pos_ = prototype_[i].repeat(x_.shape[0], 1)
        #     pos_loss += F.kl_div(x_, target_pos_, reduction='batchmean')

        #     neg_loss_ = 0
        #     for j in range(len(class_causal)):
        #         if j == i:
        #             continue
        #         else:
        #             target_neg = prototype_[j].repeat(x_.shape[0], 1)
        #             neg_loss_ += F.kl_div(x_, target_neg, reduction='batchmean')
        #     if len(class_causal) == 1:
        #         neg_loss = 10 * pos_loss
        #     else:
        #         neg_loss += neg_loss_ / (len(class_causal) - 1)

        # pos_loss /= len(class_causal)
        # neg_loss /= len(class_causal)
        # loss = args.beta * torch.exp((pos_loss - neg_loss))
        # return loss

        #infonce方式
        # l_pos = torch.einsum('nc,nc->n', [
        #     class_causal[i], prototype[i].repeat(class_causal[i].shape[0], 1)
        # ]).unsqueeze(-1)
        # negative logits: NxK

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
    # pdb.set_trace()
    return args.beta * loss
