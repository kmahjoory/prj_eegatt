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


def sep_folds(fpaths, subj_indxs, nfolds=5):
    """"""
    nfolds = nfolds
    fpath_folds = [[] for k in range(nfolds)]
    n_files = len(fpaths)
    fdirs = [os.path.dirname(k) for k in fpaths]
    fnames = [os.path.basename(k) for k in fpaths]
    cnt1, cnt2 = 0, 0
    for subjindx in subj_indxs:

        fnames_subj = [k for k in fnames if f's_{subjindx}_' in k]
        epochs_indx = []
        for fname in fnames_subj:
            indx_ = [pos for pos, c in enumerate(fname) if c == '_']
            epochs_indx.append(int(fname[indx_[2]+1:indx_[3]]))
        epochs_indx = list(set(epochs_indx))

        epochsegs_indx = {}
        for epoch in epochs_indx:
            name_pattern = f"s_{subjindx}_e_{epoch}_"
            fname_segs = [k for k in fnames_subj if name_pattern in k]
            segs_indx = []
            for fnameseg in fname_segs:
                indx_ = [pos for pos, c in enumerate(fnameseg) if c == '_']
                segs_indx.append(int(fnameseg[indx_[4]+1:indx_[5]]))
            epochsegs_indx[epoch] = sorted(segs_indx)
            foldepochseg_indxs = unidist_segs_in_folds(len(epochsegs_indx[epoch]), nfolds=5)
            foldepochsegs = [np.array(epochsegs_indx[epoch])[foldepochseg_indxs[k]].tolist() for k in range(nfolds)]
            for f, findx in enumerate(foldepochsegs):
                fname_pattern_fold = [f"s_{subjindx}_e_{epoch}_seg_{k}_" for k in findx]
                fname_fold = [os.path.join(fdirs[0], j) for k in fname_pattern_fold for j in fnames_subj if k in j]
                fpath_folds[f].extend(fname_fold)
    return fpath_folds







# Specify data directory
data_path = '/hpc/workspace/2021-0292-NeuroFlex/prj_eegatt/data_preprocessed/'
data_dir = 'data_eegatt_overlap_50_5_sec_mednorm/train/'#''data_eegatt_rl_50ol_10_sec_mednorm/train/'
path = data_path + data_dir
subj_indxs = list(range(1, 19))
subj_indxs = [k for k in subj_indxs if k not in [8, 11]]

# Separate train and test data
fpaths_all = sep_train_test(path, subj_indxs=subj_indxs, indx_test_seg=5)
train_r = fpaths_all['train_r']
train_l = fpaths_all['train_l']
test_r = fpaths_all['test_r']
test_l = fpaths_all['test_l']

folds_r = sep_folds(fpaths=train_r, subj_indxs=subj_indxs, nfolds=5)
folds_l = sep_folds(fpaths=train_l, subj_indxs=subj_indxs, nfolds=5)

# Double check there is no overlap
print([sum([k in folds_r[0] for k in folds_r[j]]) for j in range(5)])
# Double check if test and train overlap
print([sum([k in test_r for k in folds_r[j]]) for j in range(5)])
print([sum([k in test_l for k in folds_l[j]]) for j in range(5)])

print([len(folds_r[k]) for k in range(5)])
print([len(folds_l[k]) for k in range(5)])

print(sum([len(folds_l[k]) for k in range(5)]), len(train_l))
print(sum([len(folds_r[k]) for k in range(5)]), len(train_r))


if 0:
    data = [train_r+train_l, test_r+test_l]
    title_txt = ['Train', 'Test']
    #data = [folds_r[f_]+folds_l[f_] for f_ in range(5)]
    #title_txt = [f"fold{f_}" for f_ in range(1, 6)]
    ncond = len(data)

    for k in range(ncond):
        fpaths = data[k]
        fnames = [os.path.basename(k) for k in fpaths]
        fdirs = [os.path.dirname(k) for k in fpaths]
        subj_ids, epoch_indxs = [], []
        gender, acoustic, story = [], [], []
        for fname in fnames:
            indx_ = [pos for pos, c in enumerate(fname) if c == '_']
            subj_ids.append(int(fname[indx_[0] + 1:indx_[1]]))
            epoch_indxs.append(int(fname[indx_[2] + 1:indx_[3]]))
            gender.append(fname[indx_[6] + 1:indx_[7]])
            acoustic.append(int(fname[indx_[8] + 1:indx_[9]]))
            story.append(int(fname[indx_[10] + 1:indx_[11]]))


        gender_label = list(set(gender))
        gender_counts = [gender.count(k) for k in gender_label]
        if k == 0:
            fig, ax0 = plt.subplots(ncond, 1, sharex=True)
        ax0[k].bar(gender_label, gender_counts, width=.2)
        if k == ncond-1:
            ax0[k].set_xlabel('Attended Speaker Gender')
        ax0[k].set_ylabel('Counts')
        ax0[k].set_title(title_txt[k])

        story_label = list(set(story))
        story_counts = [story.count(k) for k in story_label]
        if k == 0:
            fig, ax1 = plt.subplots(ncond, 1, sharex=True)
        ax1[k].bar(story_label, story_counts)
        if k == ncond-1:
            ax1[k].set_xlabel('Attended Story')
        ax1[k].set_ylabel('Counts')
        ax1[k].set_title(title_txt[k])

        acoustic_label = list(set(acoustic))
        acoustic_counts = [acoustic.count(k) for k in acoustic_label]
        if k == 0:
            fig, ax2 = plt.subplots(ncond, 1, sharex=True)
        ax2[k].bar(acoustic_label, acoustic_counts)
        if k == ncond-1:
            ax2[k].set_xlabel('Acoustic condition')
        ax2[k].set_ylabel('Counts')
        ax2[k].set_title(title_txt[k])

        subj_ids_label = list(set(subj_ids))
        subj_ids_counts = [subj_ids.count(k) for k in subj_ids_label]
        if k == 0:
            fig, ax3 = plt.subplots(ncond, 1, sharex=True)
        ax3[k].bar(subj_ids_label, subj_ids_counts)
        if k == ncond-1:
            ax3[k].set_xlabel('Subject')
            ax3[k].set_xticks(list(range(1, 19)))
        ax3[k].set_ylabel('Counts')
        ax3[k].set_title(title_txt[k])

        epoch_indxs_label = list(set(epoch_indxs))
        epoch_indxs_counts = [epoch_indxs.count(k) for k in epoch_indxs_label]
        if k == 0:
            fig, ax4 = plt.subplots(ncond, 1, sharex=True)
        ax4[k].bar(epoch_indxs_label, epoch_indxs_counts)
        if k == ncond-1:
            ax4[k].set_xlabel('Epochs')
        ax4[k].set_ylabel('Counts')
        ax4[k].set_title(title_txt[k])

