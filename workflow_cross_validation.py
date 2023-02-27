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

from torchsummary import summary

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from eegatt_tools.models import CnnCos1layer
from eegatt_tools.train import train_model, eval_model, acc_bce_multiout, acc_bce_1out
from eegatt_tools.prepare_data import sep_train_test, sep_folds, unidist_segs_in_folds


class EEGDataset(Dataset):
    def __init__(self, sample_files, labels, transform=None, target_transform=None):
        self.sample_files = sample_files
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        sig = np.load(self.sample_files[idx], allow_pickle=True)
        eeg = torch.from_numpy(sig)
        labels = self.labels[idx]
        if self.transform:
            eeg = self.transform(sig)
        if self.target_transform:
            labels = self.target_transform(labels)
        # sample = {"eeg": sig, "label": label}
        return eeg, labels



# Specify data directory
data_path = '/hpc/workspace/2021-0292-NeuroFlex/prj_eegatt/data_preprocessed/'
data_dir = 'data_eegatt_overlap_50_5_sec_mednorm/train/'#''data_eegatt_rl_50ol_10_sec_mednorm/train/'
path = data_path + data_dir
subj_indxs = list(range(4, 6))
subj_indxs = [k for k in subj_indxs if k not in [8, 11]]

# Separate train and test data
fpaths_all = sep_train_test(path, subj_indxs=subj_indxs, indx_test_seg=5)
train_r = fpaths_all['train_r']
train_l = fpaths_all['train_l']
test_r = fpaths_all['test_r']
test_l = fpaths_all['test_l']

# Separate folds
folds_r_test, folds_r_train = sep_folds(fpaths=train_r, subj_indxs=subj_indxs, nfolds=5)
folds_l_test, folds_l_train = sep_folds(fpaths=train_l, subj_indxs=subj_indxs, nfolds=5)

# Double check there is no overlap between left and right
for ifold in range(5):
    assert not sum([sum([k in folds_r_test[ifold] for k in folds_l_test[j]]) for j in range(5)]), "Overlap between L and R"

# Double check if test and train overlap
assert not sum([sum([k in test_r for k in folds_r_test[j]]) for j in range(5)]), "Overlap between test and folds"
assert not sum([sum([k in test_l for k in folds_l_test[j]]) for j in range(5)]), "Overlap between test and folds"

print("Number of samples, Right attention: " + ' '.join([f"fold {k}: {len(folds_r_test[k])}  " for k in range(5)]))
print("Number of samples, Left attention: " + ' '.join([f"fold {k}: {len(folds_l_test[k])}  " for k in range(5)]))

# Double check if total number of samples in folds equals to the number of training samples
print(f"[L] N-samples for 5 cross-folds: {[len(folds_l_test[k] + folds_l_train[k]) for k in range(5)]}, Total number of samples: {len(train_l)}")
print(f"[R] N-samples for 5 cross-folds: {[len(folds_r_test[k] + folds_r_train[k]) for k in range(5)]}, Total number of samples: {len(train_r)}")


nfolds = 5
for fold in range(1):

    fpath_test_r = folds_r_test[fold]
    fpath_test_l = folds_l_test[fold]
    fpath_train_r = folds_r_train[fold]
    fpath_train_l = folds_l_train[fold]

    # Concatenate file paths for Right (0) and Left (1) attention
    fpath_train = fpath_train_r + fpath_train_l
    train_labels = torch.cat((torch.zeros(len(fpath_train_r)), torch.ones(len(fpath_train_l))), dim=0).to(torch.long)
    fpath_test = fpath_test_r + fpath_test_l
    test_labels = torch.cat((torch.zeros(len(fpath_test_r)), torch.ones(len(fpath_test_l))), dim=0).to(torch.long)

# Create an instance of data
sample_append = 650
sample_append_test = sample_append // 10
train_ds = EEGDataset(sample_files=fpath_train[726-sample_append:760+sample_append], labels=train_labels[726-sample_append:760+sample_append])
test_ds = EEGDataset(sample_files=fpath_test[198-sample_append_test:203+sample_append_test], labels=test_labels[198-sample_append_test:203+sample_append_test])

# Data loader
bsize = 32 # len(train_ds)
train_dl = DataLoader(train_ds, batch_size=bsize, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=True)

# GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # 0 indicates the current device
    print(torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(),
          torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')

device = torch.device('cpu')


model = CnnCos1layer().to(torch.double)
model.to(device)

n_epochs, lr, train_dl, test_dl = 200, 5*1e-4, train_dl, test_dl
loss_fn, acc_fn, device, verbose = nn.CrossEntropyLoss(), acc_bce_multiout, device, True
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5*1e-4)

best_model, metrics = train_model(train_dl, test_dl, model, loss_fn, acc_fn, optimizer, device, n_epochs=n_epochs)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
ax[0].plot(metrics['loss_train'], label='Train')
ax[0].plot(metrics['loss_test'], label='Test')
ax[0].set_xlabel('Epoch')
ax[0].legend()
ax[1].plot(metrics['loss_batches'], label='Train')
ax[1].set_xlabel('Batch')
ax[1].legend()







