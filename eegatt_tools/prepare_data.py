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
from eegatt_tools import misc


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
    fname_folds = [[] for k in range(nfolds)]
    fname_outfolds = [[] for k in range(nfolds)]
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
                fnamefold = [os.path.join(fdirs[0], j) for k in fname_pattern_fold for j in fnames_subj if k in j]
                fname_folds[f].extend(fnamefold)

                # Remove samples overlapping with test fold (Due to 50% overlap)
                outfold_idx_r = list(misc.flatten(foldepochsegs[f+1:]))
                if len(outfold_idx_r):
                    del outfold_idx_r[0]
                outfold_idx_l = list(misc.flatten(foldepochsegs[:f]))
                if len(outfold_idx_l):
                    del outfold_idx_l[-1]
                outfold_idx = outfold_idx_l + outfold_idx_r
                fname_pattern_outfold = [f"s_{subjindx}_e_{epoch}_seg_{k}_" for k in outfold_idx]
                fnameoutfold = [os.path.join(fdirs[0], j) for k in fname_pattern_outfold for j in fnames_subj if k in j]
                fname_outfolds[f].extend(fnameoutfold)

    return fname_folds, fname_outfolds


def sep_folds_deprecated(fpaths, subj_indxs, nfolds=5):
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






