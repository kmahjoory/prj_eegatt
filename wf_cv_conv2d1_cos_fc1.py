##
import os, sys, glob, json
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

from eegatt_tools.models import CnnCos1layer, weight_init, Conv2d1CosFc1, weight_init_conv2d1cosfc1
from eegatt_tools.train import train_model, eval_model, acc_bce_multiout, acc_bce_1out, reset_weights
from eegatt_tools.prepare_data import sep_train_test, sep_folds, unidist_segs_in_folds

from datetime import datetime


## Specify RUN parameters
########################################################################################################################

# Hyper-parameters for grid search
lr_searchspace = [1e-4, 1e-3, 1e-2, 1e-1]
weight_decay_searchspace = [5*1e-3]

n_epochs = 150
bsize = 64

save_output = True
########################################################################################################################


# GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # 0 indicates the current device
    print(torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(),
          torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')

device = torch.device('cpu')



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
        eeg0 = torch.from_numpy(sig).unsqueeze(0) # To make the data 3d (1channel, and and image)
        eeg = eeg0# (eeg0 - eeg0.mean()) / (eeg0.max()-eeg0.min())
        labels = self.labels[idx]
        if self.transform:
            eeg = self.transform(eeg)
        if self.target_transform:
            labels = self.target_transform(labels)
        # sample = {"eeg": sig, "label": label}
        return eeg, labels



# Specify data directory
data_path = '/hpc/workspace/2021-0292-NeuroFlex/prj_eegatt/data_preprocessed/'
data_dir = 'data_eegatt_overlap_50_5_sec_mednorm/train/'#''data_eegatt_rl_50ol_10_sec_mednorm/train/'
path = data_path + data_dir
subj_indxs = list(range(1, 5))
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





if save_output:
    datetime_txt = datetime.today().strftime('%Y%m%d_%H%M%S')
    write_path = '/hpc/workspace/2021-0292-NeuroFlex/prj_eegatt/analyses/' + f'cv_{datetime_txt}_conv2d1_cos_fc1/'
    os.mkdir(write_path)

learning_rates = []
weight_decays = []
accs_mean = []

for lr in lr_searchspace:
    for wd in weight_decay_searchspace:


        model = Conv2d1CosFc1()
        loss_fn, acc_fn, device, verbose = nn.CrossEntropyLoss(), acc_bce_multiout, device, True
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        #optimizer = optim.SGD([{"params": model.conv1.parameters(), 'lr': lr, "momentum": 0.9},
        #                       {"params": model.fc.parameters(), 'lr': lr, "momentum": 0.9}],
        #                      lr=lr, momentum=0.9, weight_decay=wd)

        # Training
        ##############################################################################################################
        nfolds = 5
        accs_fold = []
        metrics_fold = []
        for fold in range(nfolds):

            # model.apply(reset_weights)
            model = weight_init_conv2d1cosfc1(model)
            model.to(torch.double)
            model.to(device)

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
            train_ds = EEGDataset(sample_files=fpath_train, labels=train_labels)
            test_ds = EEGDataset(sample_files=fpath_test, labels=test_labels)

            # Data loader
            train_dl = DataLoader(train_ds, batch_size=bsize, shuffle=True)
            test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=True)

            # Training
            best_model, metrics = train_model(train_dl, test_dl, model, loss_fn, acc_fn, optimizer, device, n_epochs=n_epochs)

            # Save output
            metrics_fold.append(metrics)
            best_model_weights = best_model.state_dict().copy()
            if save_output:
                torch.save(best_model_weights, os.path.join(write_path, f"bestmodel_lr_{lr}_wd_{wd}_fold_{fold}.pth"))
            accs_fold.append(metrics['best_acc'])

        if save_output:
            # Save metrics from folds
            write_dir_foldmtrcs = os.path.join(write_path, f'metrics_fold_lr_{lr}_wd_{wd}.json')
            with open(write_dir_foldmtrcs, 'w') as f:
                json.dump(metrics_fold, f)

        accs_mean.append(sum(accs_fold)/len(accs_fold))
        weight_decays.append(wd)
        learning_rates.append(lr)

if save_output:
    df = pd.DataFrame({'acc_mean': accs_mean, 'weight_decay': weight_decays, 'learning_rate': learning_rates})
    df.to_csv(os.path.join(write_path, 'mean_accuracy.csv'))
