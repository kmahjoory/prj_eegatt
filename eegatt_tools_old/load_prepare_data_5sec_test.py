##
import os, sys, glob
import math, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn  # Neural network module
import torch.nn.functional as F  # Neural network module
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from eegatt_tools.utils import tensor_info, dl_info
from eegatt_tools.model_utils import CNN1d1_fc1, CNN1d1_fc2, CNN1d2_fc1, CNN1d2_fc2, CNN1d1_h2_fc1, CNN1d1_h2_fc2
from eegatt_tools.train_utils import train_model, initialize_weights, acc_bce_multiout, acc_bce_1out, eval_model, \
    rep_runs, run_multimodel_exps, run_multilr_exps, plot_scores
from eegatt_tools.sig_vis_utils import plot_multichannel


def sep_train_test(path, subj_indxs=[2, 3, 4], indx_test_seg=5):
    fpaths_all_train_r, fpaths_all_test_r = [], []
    fpaths_all_train_l, fpaths_all_test_l = [], []
    for subjindx in subj_indxs:
        # Right attention
        fpaths_r = set(glob.glob(f'{path}r/s_{subjindx}_*.npy'))
        fpaths_test_r = set(glob.glob(f'{path}r/s_{subjindx}_*seg_{indx_test_seg}*.npy'))
        fpaths_rm_r = set.union(set(glob.glob(f'{path}r/s_{subjindx}_*seg_{indx_test_seg - 1}*.npy')),
                                set(glob.glob(f'{path}r/s_{subjindx}_*seg_{indx_test_seg + 1}*.npy')))
        fpaths_train_r = list(fpaths_r - fpaths_test_r - fpaths_rm_r)
        fpaths_test_r = list(fpaths_test_r)
        # Put file paths for all subjects in a list
        fpaths_all_train_r.extend(fpaths_train_r)
        fpaths_all_test_r.extend(fpaths_test_r)

        # Left attention
        fpaths_l = set(glob.glob(f'{path}l/s_{subjindx}_*.npy'))
        fpaths_test_l = set(glob.glob(f'{path}l/s_{subjindx}_*seg_{indx_test_seg}*.npy'))
        fpaths_rm_l = set.union(set(glob.glob(f'{path}l/s_{subjindx}_*seg_{indx_test_seg - 1}*.npy')),
                                set(glob.glob(
                                    f'{path}l/s_{subjindx}_*seg_{indx_test_seg + 1}*.npy')))
        fpaths_train_l = list(fpaths_l - fpaths_test_l - fpaths_rm_l)
        fpaths_test_l = list(fpaths_test_l)
        # Put file paths for all subjects in a list
        fpaths_all_train_l.extend(fpaths_train_l)
        fpaths_all_test_l.extend(fpaths_test_l)
    return {'train_r': fpaths_all_train_r, 'train_l': fpaths_all_train_l, 'test_r': fpaths_all_test_r,
            'test_l': fpaths_all_test_l}


def unidist_segs_in_folds(nsegs, nfolds=5):
    import random
    folds = [nsegs // nfolds] * nfolds
    for k in range(nsegs % nfolds):
        indx = random.randint(1, nfolds)
        folds[indx - 1] += 1
    folds_slice = [sum(folds[:k]) for k in range(1, len(folds) + 1)]
    folds_slice.insert(0, 0)
    foldsegs_indx = [list(range(folds_slice[k], folds_slice[k + 1])) for k in range(5)]
    return foldsegs_indx


def load_prepare_data_5sec(indx_subj, indx_test_seg=5, indx_subgroup='*'):
    """"""

    # some initial setup
    #np.set_printoptions(precision=2)
    #use_gpu = torch.cuda.is_available()
    #np.random.seed(1234)

    data_path = '/hpc/workspace/2021-0292-NeuroFlex/prj_eegatt/data_preprocessed/'
    data_dir = 'data_eegatt_overlap_50_5_sec_mednorm/train/'#''data_eegatt_rl_50ol_10_sec_mednorm/train/'
    path = data_path + data_dir



    subj_indxs = [2, 3]
    f = sep_train_test(path, subj_indxs=subj_indxs, indx_test_seg=5)
    #def sep_folds(fpahts, subj_indxs)
    fpath_folds = [[] for k in range(nfolds)]
    ffpaths = f['train_r']
    for subjindx in subj_indxs:
        fdirs = [os.path.dirname(k) for k in ffpaths]
        fnames = [os.path.basename(k) for k in ffpaths]

        n_files = len(r_fpath_train)
        fpaths = [os.path.dirname(k) for k in r_fpath_train]
        fnames = [os.path.basename(k) for k in r_fpath_train]

        epochs_indx = []
        for fname in fnames:
            indx_ = [pos for pos, c in enumerate(fname) if c == '_']
            epochs_indx.append(int(fname[indx_[2] + 1:indx_[3]]))
        epochs_indx = list(set(epochs_indx))

        epochsegs_indx = {}
        for epoch in epochs_indx:
            name_pattern = f"s_{subjindx}_e_{epoch}_seg*"
            fpath_segs = glob.glob(os.path.join(fdirs[0], name_pattern))
            segs_indx = []
            for fpathseg in fpath_segs:
                fnameseg = os.path.basename(fpathseg)
                indx_ = [pos for pos, c in enumerate(fnameseg) if c == '_']
                segs_indx.append(int(fnameseg[indx_[4]+1:indx_[5]]))
            epochsegs_indx[epoch] = sorted(segs_indx)
            foldepochseg_indxs = unidist_segs_in_folds(len(epochsegs_indx[epoch]), nfolds=5)
            for f, findx in enumerate(foldepochseg_indxs):
                fpath_folds[f].append([f"s_{subjindx}_e_{epoch}_seg_{k}" for k in findx])













            seg_indx = int(fname[indx_[4] + 1:indx_[5]])

        fnames_indxs = []
        for fname in fnames:
            indx_ = [pos for pos, c in enumerate(fname) if c == '_']
            subj_indx = int(fname[indx_[0] + 1:indx_[1]])
            epochs_indx = int(fname[indx_[2] + 1:indx_[3]])
            seg_indx = int(fname[indx_[4] + 1:indx_[5]])




            fpaths_l.append(glob.glob(f'{path}l/s_{subjindx}_*.npy'))


        r_fpath_rm = set.union(set(glob.glob(f'{path}r/s_{indx_subj}_*seg_{indx_test_seg-1}{indx_subgroup}.npy')),
                               set(glob.glob(f'{data_path}{data_dir}r/s_{indx_subj}_*seg_{indx_test_seg+1}{indx_subgroup}.npy')))
        r_fpath_train = list(r_fpath - r_fpath_test - r_fpath_rm)
        r_fpath_test = list(r_fpath_test)
        return {'r': fpaths_r, 'l': fpaths_l }

    def sep_train_test(fpaths, test_seg_indx=5):

        path = os.path.dirname(fpaths[0])
        r_fpath, l_fpath = fpaths['r'], fpaths['l']



    fpaths = pick_subjects(path)
    sep_train_test(path)
        r_fpath = set(glob.glob(f'{path}r/s_{indx_subj}_{indx_subgroup}.npy'))
        indx_subgroup = '*'
        r_fpath = set(glob.glob(f'{path}r/s_{indx_subj}_{indx_subgroup}.npy'))
        r_fpath_test = set(glob.glob(f'{path}r/s_{indx_subj}_*seg_{indx_test_seg}{indx_subgroup}.npy'))
        r_fpath_rm = set.union(set(glob.glob(f'{path}r/s_{indx_subj}_*seg_{indx_test_seg-1}{indx_subgroup}.npy')),
                               set(glob.glob(f'{data_path}{data_dir}r/s_{indx_subj}_*seg_{indx_test_seg+1}{indx_subgroup}.npy')))
        r_fpath_train = list(r_fpath - r_fpath_test - r_fpath_rm)
        r_fpath_test = list(r_fpath_test)

        l_fpath = set(glob.glob(f'{path}l/s_{indx_subj}_{indx_subgroup}.npy'))
        l_fpath_test = set(glob.glob(f'{path}l/s_{indx_subj}_*seg_{indx_test_seg}{indx_subgroup}.npy'))
        l_fpath_rm = set.union(
            set(glob.glob(f'{data_path}{data_dir}l/s_{indx_subj}_*seg_{indx_test_seg - 1}{indx_subgroup}.npy')),
            set(glob.glob(f'{data_path}{data_dir}l/s_{indx_subj}_*seg_{indx_test_seg + 1}{indx_subgroup}.npy')))
        l_fpath_train = list(l_fpath - l_fpath_test - l_fpath_rm)
        l_fpath_test = list(l_fpath_test)
        return {'fpaths_train_r': r_fpath_train, 'fpaths_test_r': r_fpath_test,'fpaths_train_l': l_fpath_train, 'fpaths_test_l': l_fpath_test}

    subj_indxs = [2, 3, 4]
    for subjindx in subj_indxs:
        fpaths = sep_train_test(path, indx_subj, indx_test_seg=5)


    n_files = len(r_fpath_train)
    fpaths = [os.path.dirname(k) for k in r_fpath_train]
    fnames = [os.path.basename(k) for k in r_fpath_train]

    fnames_indxs = []
    for fname in fnames:
        indx_ = [pos for pos, c in enumerate(fname) if c == '_']
        subj_indx = int(fname[indx_[0]+1:indx_[1]])
        epochs_indx = int(fname[indx_[2]+1:indx_[3]])
        seg_indx = int(fname[indx_[4]+1:indx_[5]])
        fnames_indxs.append([subj_indx, epoch_indx, seg_indx])














    l_fpath = set(glob.glob(f'{data_path}{data_dir}l/s_{indx_subj}_{indx_subgroup}.npy'))
    l_fpath_test = set(glob.glob(f'{data_path}{data_dir}l/s_{indx_subj}_*seg_{indx_test_seg}{indx_subgroup}.npy'))
    l_fpath_rm = set.union(set(glob.glob(f'{data_path}{data_dir}l/s_{indx_subj}_*seg_{indx_test_seg-1}{indx_subgroup}.npy')), set(glob.glob(f'{data_path}{data_dir}l/s_{indx_subj}_*seg_{indx_test_seg+1}{indx_subgroup}.npy')))
    l_fpath_train = list(l_fpath - l_fpath_test - l_fpath_rm)
    l_fpath_test = list(l_fpath_test)

    train_files = l_fpath_train + r_fpath_train
    train_labels = torch.cat((torch.zeros(len(l_fpath_train)), torch.ones(len(r_fpath_train))), dim=0).to(torch.long)

    test_files = l_fpath_test + r_fpath_test
    test_labels = torch.cat((torch.zeros(len(l_fpath_test)), torch.ones(len(r_fpath_test))), dim=0).to(torch.long)

    print(f"Ntrain={len(train_files)},  {len(test_files)}")

    class EEGDataset(Dataset):
        def __init__(self, sample_files, labels, transform=None, target_transform=None):
            self.sample_files = sample_files
            self.labels = labels
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            #import pdb; pdb.set_trace()
            sig = np.load(self.sample_files[idx], allow_pickle=True)
            sig = torch.from_numpy(sig)
            label = self.labels[idx]
            if self.transform:
                sig = self.transform(sig)
            if self.target_transform:
                label = self.target_transform(label)
            #sample = {"eeg": sig, "label": label}
            return sig, label

    train_ds = EEGDataset(sample_files=train_files, labels=train_labels)
    test_ds = EEGDataset(sample_files=test_files, labels=test_labels)

    return train_ds, test_ds, train_files, test_files

