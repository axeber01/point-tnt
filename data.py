#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch


def translate_pointcloud(pointcloud, low1=2/3., high1=3/2., low2=-0.2, high2=0.2):
    xyz1 = np.random.uniform(low=low1, high=high1, size=[3])
    xyz2 = np.random.uniform(low=low2, high=high2, size=[3])

    translated_pointcloud = np.add(np.multiply(
        pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def download_scanobjectnn():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
        www = 'http://103.24.77.34/scanobjectnn/h5_files.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_scanobjectnn_data(partition):
    download_scanobjectnn()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_mask = []

    h5_name = BASE_DIR + '/data/h5_files/main_split/' + \
        partition + '_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    mask = f['mask'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_mask.append(mask)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_mask = np.concatenate(all_mask, axis=0)
    return all_data, all_label, all_mask


def convert_to_binary_mask(masks):
    binary_masks = []
    for i in range(masks.shape[0]):
        binary_mask = np.ones(masks[i].shape)
        bg_idx = np.where(masks[i, :] == -1)
        binary_mask[bg_idx] = 0

        binary_masks.append(binary_mask)

    binary_masks = np.array(binary_masks)
    return binary_masks


class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='training', mask=False):
        self.data, self.label, self.mask = load_scanobjectnn_data(partition)
        self.num_points = num_points
        self.partition = partition

        self.mask = convert_to_binary_mask(self.mask)
        self.return_mask = mask

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        mask = self.mask[item][:self.num_points]

        if self.partition == 'training':
            pointcloud = translate_pointcloud(pointcloud)

        if self.return_mask:
            return pointcloud, (label, mask)
        else:
            return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)

    train = ScanObjectNN(1024)
    test = ScanObjectNN(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
