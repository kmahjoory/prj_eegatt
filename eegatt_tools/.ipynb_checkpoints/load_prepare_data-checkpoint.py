
def load_prepare_data(n_samples4class=5):
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
    data_dir = 'data_eegatt_rl_50ol_10_sec_mednorm/train/'

    r_fpath = list(glob.glob(f'{data_path}{data_dir}r/s_18_*gen_m_ac_1.npy'))[:n_samples4class]
    r_labels = list([1]*len(r_fpath))
    l_fpath = list(glob.glob(f'{data_path}{data_dir}l/s_18_*gen_m_ac_1.npy'))[:n_samples4class]
    l_labels = list([0]*len(l_fpath))

    files = l_fpath + r_fpath
    labels = torch.cat((torch.zeros(len(l_fpath)), torch.ones(len(r_fpath))), dim=0).to(torch.long)


    print(f"L-Att files: N={len(l_fpath)},  {l_fpath[:5]}")
    print(f"R-Att files: N={len(r_fpath)},  {r_fpath[:5]}")


    indx_r = np.arange(len(r_fpath))
    np.random.shuffle(indx_r)
    indx_l = np.arange(len(l_fpath))
    np.random.shuffle(indx_l)

    n_test_r = int(.2*indx_r.shape[0])
    n_test_l = int(.2*indx_l.shape[0])

    indx_test_r = indx_r[:n_test_r]
    indx_test_l = indx_l[:n_test_l]
    indx_train_r = indx_r[n_test_r:]
    indx_train_l = indx_l[n_test_l:]


    indx_test_r = np.random.randint(len(r_fpath), size=int(.2*len(r_fpath)))     #[6, 15] #[2, 6, 15, 22, 28]
    indx_test_l = np.random.randint(len(l_fpath), size=int(.2*len(l_fpath)))
    indx_train_r = np.setdiff1d(np.arange(len(r_fpath)), indx_test_r)
    indx_train_l = np.setdiff1d(np.arange(len(l_fpath)), indx_test_l)

    train_r_fpath = [r_fpath[j] for j in indx_train_r]
    train_l_fpath = [l_fpath[j] for j in indx_train_l]
    test_r_fpath = [r_fpath[j] for j in indx_test_r]
    test_l_fpath = [l_fpath[j] for j in indx_test_l]


    print(f"N-R: {len(r_fpath)},   N-R-Train: {len(train_r_fpath)},   N-R-Test: {len(test_r_fpath)}")
    print(f"N-L: {len(l_fpath)},   N-L-Train: {len(train_l_fpath)},   N-L-Test: {len(test_l_fpath)}")


    train_fpath = train_l_fpath + train_r_fpath
    train_labels = torch.cat((torch.zeros(len(train_l_fpath)), torch.ones(len(train_r_fpath))), dim=0).to(torch.long)

    test_fpath = test_l_fpath + test_r_fpath
    test_labels = torch.cat((torch.zeros(len(test_l_fpath)), torch.ones(len(test_r_fpath))), dim=0).to(torch.long)


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

    train_ds = EEGDataset(sample_files=train_fpath, labels=train_labels)
    test_ds = EEGDataset(sample_files=train_fpath, labels=train_labels)

    return train_ds, test_ds

