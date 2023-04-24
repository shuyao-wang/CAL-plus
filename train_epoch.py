import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import numpy as np

import time
import random
import pdb


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train_causal_epoch(model, optimizer, loader, device, memory_bank,
                       prototype, criterion, args):

    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_p = 0
    total_loss_co = 0

    for step, data in enumerate(loader):

        optimizer.zero_grad()
        data = data.to(device)

        if data.x.shape[0] == 1:
            pass
        else:
            if args.memory == True:
                start = time.time()
                # k = 0
                for step_, data_ in enumerate(loader):

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

            if args.prototype == True:
                class_causal = class_split(data.y, xo, args.num_classes)
                p_loss = loss_prototype(prototype, class_causal, criterion,
                                        args)
                p_loss.backward(retain_graph=True)
            else:
                p_loss = 0

            c_logs, o_logs, co_logs = model.forward(xo, xc)
            one_hot_target = data.y.view(-1).type(torch.int64)
            target_rep = data.y.to(torch.float).repeat_interleave(
                data.batch[-1] + 1, dim=0).view(-1).type(torch.int64)
            uniform_target = torch.ones_like(
                c_logs, dtype=torch.float).to(device) / args.num_classes
            o_loss = F.nll_loss(o_logs, one_hot_target)
            co_loss = F.nll_loss(co_logs, target_rep)
            c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')

            loss = args.c * c_loss + args.o * o_loss + args.co * co_loss
            loss.backward()

            total_loss += (loss.item() + p_loss) * num_graphs(data)
            total_loss_c += c_loss.item() * num_graphs(data)
            total_loss_o += o_loss.item() * num_graphs(data)
            total_loss_p += p_loss * num_graphs(data)
            total_loss_co += co_loss.item() * num_graphs(data)
            optimizer.step()

    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_p = total_loss_p / num
    total_loss_co = total_loss_co / num
    # correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, total_loss_p,


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
    return args.beta * loss