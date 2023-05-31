# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import copy
import torch
from torch_geometric.data import Data, Batch
from utils.misc import BlackHole
from easydict import EasyDict


def repeat_data(data: Data, num_repeat) -> Batch:
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    return Batch.from_data_list(datas)


def repeat_batch(batch: Batch, num_repeat) -> Batch:
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss


def log_losses(out, it, tag, train_report_iter=1, logger=BlackHole(), writer=BlackHole(), others={}):
    if it % train_report_iter == 0:
        logstr = '[%s] Iter %05d' % (tag, it)
        logstr += ' | loss %.4f' % out['overall'].item()
        for k, v in out.items():
            if k == 'overall': continue
            logstr += ' | loss(%s) %.4f' % (k, v.item())
        for k, v in others.items():
            if k == 'lr':
                logstr += ' | %s %2.6f' % (k, v)
            else:
                logstr += ' | %s %2.4f' % (k, v)
        logger.info(logstr)

    for k, v in out.items():
        if k == 'overall':
            writer.add_scalar('%s/loss' % tag, v, it)
        else:
            writer.add_scalar('%s/loss_%s' % (tag, k), v, it)
    for k, v in others.items():
        writer.add_scalar('%s/%s' % (tag, k), v, it)
    writer.flush()


class ValidationLossTape(object):

    def __init__(self):
        super().__init__()
        self.accumulate = {}
        self.others = {}
        self.total = 0

    def update(self, out, n, others={}):
        self.total += n
        for k, v in out.items():
            if k not in self.accumulate:
                self.accumulate[k] = v.clone().detach()
            else:
                self.accumulate[k] += v.clone().detach()

        for k, v in others.items():
            if k not in self.others:
                self.others[k] = v.clone().detach()
            else:
                self.others[k] += v.clone().detach()

    def log(self, it, logger=BlackHole(), writer=BlackHole(), tag='val', others={}):
        avg = EasyDict({k: v / self.total for k, v in self.accumulate.items()})
        avg_others = EasyDict({k: v / self.total for k, v in self.others.items()})
        avg_others.update(others)
        log_losses(avg, it, tag, logger=logger, writer=writer, others=avg_others)
        return avg['overall']
