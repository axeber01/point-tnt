#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Point-TnT:
@Author: Axel Berg
@Contact: axel.berg@arm.com
@File: main.py
@Time: 2022

RSMix:
@Author: Dogyoon Lee
@Contact: dogyoonlee@gmail.com
@File: main.py
@Time: 2020/11/23 13:46 PM

DGCNN:
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
'''


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ScanObjectNN
from model import PointTNT, Baseline
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from torchsummary import summary
import time
from datetime import datetime
import provider
import rsmix_provider
from warmup_scheduler import GradualWarmupScheduler


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp -r *.py checkpoints/'+args.exp_name+'/')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def train(args, io):
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=args.num_data_workers,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=args.num_data_workers,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class = 40

    elif args.dataset == 'scanobjectnn':
        train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points), num_workers=args.num_data_workers,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=args.num_data_workers,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class = 15

    device = torch.device('cuda' if args.cuda else 'cpu')

    # load model
    if args.model == 'point-tnt':
        model = PointTNT(args, num_class).to(device)
    elif args.model == 'baseline':
        model = Baseline(args, num_class).to(device)
    else:
        raise Exception('Model not implemented')

    summary(model, (3, args.num_points))
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs")

    # Apply weight decay to all layers, except biases, LayerNorm and BatchNorm
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'BatchNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wd},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # select optimizer
    if args.use_sgd:
        print('Use SGD')
        opt = optim.SGD(optimizer_grouped_parameters,
                        lr=args.lr, momentum=args.momentum)
    else:
        print('Use AdamW')
        opt = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    # use cosine annealing learning rate scheduler with warmup
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr/100.0)
    scheduler_warmup = GradualWarmupScheduler(
        opt, multiplier=1, total_epoch=10, after_scheduler=scheduler)

    def my_loss(pred, gold):
        return cal_loss(pred, gold, eps=args.smooth)

    criterion = my_loss

    best_test_acc = 0
    best_avg_class_acc = 0
    conv_epoch = 0
    print('Starting model training')
    for epoch in range(args.epochs):
        scheduler_warmup.step()
        log_string(str(datetime.now()))
        log_string('**** EPOCH %03d ****' % (epoch))
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            '''
            RSMIX Augmentation, inhereted from
            https://github.com/dogyoonlee/RSMix
            '''
            rsmix = False
            if args.rot or args.rdscale or args.shift or args.jitter or args.shuffle or args.rddrop or (args.beta is not 0.0):
                data = data.cpu().numpy()
            if args.rot:
                data = provider.rotate_point_cloud(data)
                data = provider.rotate_perturbation_point_cloud(data)
            if args.rdscale:
                tmp_data = provider.random_scale_point_cloud(data[:, :, 0:3])
                data[:, :, 0:3] = tmp_data
            if args.shift:
                tmp_data = provider.shift_point_cloud(data[:, :, 0:3])
                data[:, :, 0:3] = tmp_data
            if args.jitter:
                tmp_data = provider.jitter_point_cloud(data[:, :, 0:3])
                data[:, :, 0:3] = tmp_data
            if args.rddrop:
                data = provider.random_point_dropout(data)
            if args.shuffle:
                data = provider.shuffle_points(data)
            r = np.random.rand(1)
            if args.beta > 0 and r < args.rsmix_prob:
                rsmix = True
                data, lam, label, label_b = rsmix_provider.rsmix(
                    data, label, beta=args.beta, n_sample=args.nsample, KNN=args.knn)
            if args.rot or args.rdscale or args.shift or args.jitter or args.shuffle or args.rddrop or (args.beta is not 0.0):
                data = torch.FloatTensor(data)
            if rsmix:
                lam = torch.FloatTensor(lam)
                if args.dataset == 'scanobjectnn':
                    label_b = torch.FloatTensor(label_b)
                    label = torch.FloatTensor(label)
                lam, label_b = lam.to(device), label_b.to(device).squeeze()
            data, label = data.to(device), label.to(device).squeeze()

            if rsmix:
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opt.zero_grad()
                logits = model(data)

                loss = 0
                for i in range(batch_size):
                    loss_tmp = criterion(logits[i].unsqueeze(0), label[i].unsqueeze(0).long())*(1-lam[i]) \
                        + criterion(logits[i].unsqueeze(0),
                                    label_b[i].unsqueeze(0).long())*lam[i]
                    loss += loss_tmp
                loss = loss/batch_size

            else:
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opt.zero_grad()
                logits = model(data)

                loss = criterion(logits, label)

            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.detach().item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        outstr = 'Train epoch %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (
                epoch,
                train_loss*1.0/count,
                train_acc,
                metrics.balanced_accuracy_score(train_true, train_pred))

        io.cprint(outstr)
        LOG_FOUT.write(outstr+'\n')
        LOG_FOUT.flush()

        ####################
        # Test
        ####################
        log_string('---- EPOCH %03d EVALUATION ----' % (epoch))

        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            with torch.no_grad():
                logits = model(data)
                loss = criterion(logits, label)

            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.detach().item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(
            test_true, test_pred)
        outstr = 'Test epoch %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (
                epoch,
                test_loss*1.0/count,
                test_acc,
                avg_per_class_acc)

        io.cprint(outstr)

        LOG_FOUT.write(outstr+'\n')
        LOG_FOUT.flush()

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            conv_epoch = epoch
            torch.save(model.state_dict(),
                       'checkpoints/%s/models/model_best.t7' % args.exp_name)
            log_string(
                'Model saved in file : checkpoints/%s/models/model_best.t7' % (args.exp_name))
            best_avg_class_acc = avg_per_class_acc

        log_string('*** best accuracy *** - %f' % (best_test_acc))
        log_string('*** at then, best class accuracy *** -  %f' %
                   (best_avg_class_acc))

    execution_time = time.time()-start_time
    hour = execution_time//3600
    minute = (execution_time-hour*3600)//60
    second = execution_time-hour*3600-minute*60
    log_string('... End of the Training ...')
    log_string('trainig time : %.2f sec, %d min, %d hour' %
               (float(second), int(minute), int(hour)))
    log_string('*** training accuracy when best accuracy *** - %f' %
               (train_acc))
    log_string('*** best accuracy *** - %f' % (best_test_acc))
    log_string('*** at then, best class accuracy *** -  %f' %
               (best_avg_class_acc))
    log_string('*** conv epoch *** - %d' % (conv_epoch))
    torch.save(model.state_dict(),
               'checkpoints/%s/models/model_final.t7' % args.exp_name)
    log_string(
        'Final model saved in file : checkpoints/%s/models/model_final.t7' % (args.exp_name))


def test(args, io):

    if args.dataset == 'modelnet40':
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=args.num_data_workers,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class = 40

    elif args.dataset == 'scanobjectnn':
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=args.num_data_workers,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class = 15

    device = torch.device('cuda' if args.cuda else 'cpu')

    # load model
    if args.model == 'point-tnt':
        model = PointTNT(args, num_class).to(device)
    elif args.model == 'baseline':
        model = Baseline(args, num_class).to(device)
    else:
        raise Exception('Model not implemented')

    summary(model, (3, args.num_points))
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.eval()
    test_acc = 0.0
    test_true = []
    test_pred = []
    print('Starting model evalutation')
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (
        test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='point-tnt',
                        choices=['point-tnt', 'baseline'],
                        help='Model to use, [point-tnt, baseline]')
    parser.add_argument('--dataset', type=str, default='scanobjectnn',
                        choices=['scanobjectnn', 'modelnet40'])
    parser.add_argument('--num_data_workers', type=int, default=8,
                        help='The number of workers in dataloader')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='Test batch size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD, if not use AdamW')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.1,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--model_path', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--smooth', type=float, default=0.2,
                        help='Label smoothing epsilon')
    # RSMIX arguments
    parser.add_argument('--rdscale', action='store_true',
                        help='random scaling data augmentation')
    parser.add_argument('--shift', action='store_true',
                        help='random shift data augmentation')
    parser.add_argument('--shuffle', action='store_true',
                        help='random shuffle data augmentation')
    parser.add_argument('--rot', action='store_true',
                        help='random rotation augmentation')
    parser.add_argument('--jitter', action='store_true',
                        help='jitter augmentation')
    parser.add_argument('--rddrop', action='store_true',
                        help='random point drop data augmentation')
    parser.add_argument('--rsmix_prob', type=float,
                        default=0.5, help='rsmix probability')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='scalar value for beta function')
    parser.add_argument('--nsample', type=int, default=512,
                        help='default max sample number of the erased or added points in rsmix')
    parser.add_argument('--normal', action='store_true', help='use normal')
    parser.add_argument('--knn', action='store_true',
                        help='use knn instead ball-query function')

    # PointTNT arguments
    parser.add_argument('--point_dim', type=int, default=32,
                        help='Point embedding dimension')
    parser.add_argument('--patch_dim', type=int, default=192,
                        help='Anchor embedding dimension')
    parser.add_argument('--depth', type=int, default=4,
                        help='Number of sequential transformers')
    parser.add_argument('--heads', type=int, default=3,
                        help='Number of attention heads')
    parser.add_argument('--dim_head', type=int, default=64,
                        help='Attention head dimension')
    parser.add_argument('--ff_dropout', type=float, default=0.,
                        help='Feed-forward dropout')
    parser.add_argument('--attn_dropout', type=float, default=0.,
                        help='Attention dropout')
    parser.add_argument('--local_attention', type=str2bool, default=True,
                        help='Toggle the use of local attention (True/False)')
    parser.add_argument('--global_attention', type=str2bool, default=True,
                        help='Toggle the use of global attention (True/False)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024,
                        help='Dimension of embeddings')
    parser.add_argument('--n_anchor', type=int, default=192,
                        help='Number of anchor points')
    parser.add_argument('--k', type=int, default=20,
                        help='Num of nearest neighbors to use')
    parser.add_argument('--dilation', type=int, default=1,
                        help='knn dilation')

    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if not os.path.exists('./log'):
        os.mkdir('./log')
    LOG_DIR = os.path.join('./log', args.exp_name)
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(args)+'\n')

    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    start_time = time.time()

    if not args.eval:
        print('Training')
        train(args, io)
    else:
        print('Testing')
        test(args, io)

    LOG_FOUT.close()
