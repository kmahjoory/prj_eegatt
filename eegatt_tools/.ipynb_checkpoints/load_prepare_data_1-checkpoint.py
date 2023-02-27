
def load_prepare_data_1(indx_subj):
    ##
    import os, sys, glob
    import math, time

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn # Neural network module
    import torch.nn.functional as F # Neural network module
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader


    import torchvision
    import torchvision.datasets as datasets
    import torchvision.models as models
    import torchvision.transforms as transforms

    from eegatt_tools.utils import tensor_info, dl_info
    from eegatt_tools.model_utils import CNN1d1_fc1, CNN1d1_fc2, CNN1d2_fc1, CNN1d2_fc2, CNN1d1_h2_fc1, CNN1d1_h2_fc2
    from eegatt_tools.train_utils import train_model, initialize_weights, acc_bce_multiout, acc_bce_1out, eval_model, rep_runs, run_multimodel_exps, run_multilr_exps, plot_scores
    from eegatt_tools.sig_vis_utils import plot_multichannel

    # some initial setup
    #np.set_printoptions(precision=2)
    #use_gpu = torch.cuda.is_available()
    #np.random.seed(1234)

    data_path = '../eeg/data_preprocessed/'
    data_dir = 'data_eegatt_overlap_50_10_sec_mednorm/train/'#''data_eegatt_rl_50ol_10_sec_mednorm/train/'

    r_fpath = set(glob.glob(f'{data_path}{data_dir}r/s_{indx_subj}_*.npy'))
    r_fpath_test = set(glob.glob(f'{data_path}{data_dir}r/s_{indx_subj}_*seg_5*.npy'))
    r_fpath_train = list(r_fpath - r_fpath_test)
    r_fpath_test = list(r_fpath_test)

    l_fpath = set(glob.glob(f'{data_path}{data_dir}l/s_{indx_subj}_*.npy'))
    l_fpath_test = set(glob.glob(f'{data_path}{data_dir}l/s_{indx_subj}_*seg_5*.npy'))
    l_fpath_train = list(l_fpath - l_fpath_test)
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

    return train_ds, test_ds

