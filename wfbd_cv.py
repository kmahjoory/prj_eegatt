

import os, sys, glob
import math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import pdb
import scipy
from braindecode.datasets import create_from_X_y

import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
from skorch.helper import SliceDataset

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True


# ######################################################################################################################
# PREPARE DATA

subj_id = 1
data_dir = '/hpc/workspace/2021-0292-NeuroFlex/prj_eegatt/data/eeg_preprocessed/'

# Load event info and delete 1-speaker blocks
subj_event_name = f'subj_{subj_id}_expinfo.csv'
event_file_path = os.path.join(data_dir, subj_event_name)
events_info = pd.read_csv(event_file_path, index_col=0)
indx_2speakers = events_info['n_speakers'] == 2
events_info = events_info[indx_2speakers].reset_index(drop=True)
# Load EEG data and update according to event info
subj_eeg_name = f'subj_{subj_id}_eeg_after_ica.fif'
eeg_file_path = os.path.join(data_dir, subj_eeg_name)
eeg_epochs = mne.read_epochs(eeg_file_path).pick_types(eeg=True)[indx_2speakers]
sfreq = eeg_epochs.info['sfreq']
ch_names = eeg_epochs.info['ch_names']
# Preprocess EEG
eeg_epochs.filter(4, 32)
eeg = eeg_epochs.get_data()
X = eeg * 1e3
y = (events_info['attend_lr'] == 'r').astype('int').tolist()  # Right attention is labeled 1

# Windowing according to Skorch and Braindecode:
dataset = create_from_X_y(
    X, y,
    drop_last_window=False,
    sfreq=sfreq,
    ch_names=ch_names,
    window_stride_samples=int(5*sfreq),
    window_size_samples=int(5*sfreq),
)
dataset.set_description(events_info)


# ######################################################################################################################
# SEPARATE TEST & TRAIN DATA

indx_test = [10, 15, 21, 29, 49, 55]
split = dataset.split({'train': np.delete(np.arange(60), indx_test), 'test': indx_test})
train_set = split['train']
valid_set = split['test']

train_X = SliceDataset(train_set, idx=0)
train_y = train_set.get_metadata().target
valid_X = SliceDataset(valid_set, idx=0)
valid_y = valid_set.get_metadata().target

########################################################################################################################
# MODEL

#seed = 20200220
#set_random_seeds(seed=seed)
n_classes = 2
n_chans = len(ch_names)
input_window_samples = train_set[0][0].shape[1]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

########################################################################################################################
# CLASSIFIER SPECIFICATION

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier

lr = 0.03 * 0.01
weight_decay = 0.9 * 0.001
batch_size = 64
n_epochs = 20

clf = EEGClassifier(
    module=model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    iterator_train__shuffle=True,
    callbacks=[
        "accuracy",
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied in the dataset.
#clf.fit(train_set, y=None, epochs=n_epochs)

#(clf.predict(train_X) == np.array(train_y)).mean()
#(clf.predict(valid_X) == np.array(valid_y)).mean()
########################################################################################################################
# Stratified Kfold cross-validation

from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5)

train_X = SliceDataset(train_set, idx=0)
train_y = np.array(train_set.get_metadata().target)


#for params_grid
for indx_train, indx_valid in kf.split(train_set):
    fit_params = {'optimizer__lr': 0.001, 'optimizer__weight_decay': 0.001}
    clf.train_split = predefined_split(valid_set)
    clf.fit(train_X[indx_train], train_y[indx_train])



param_grid = {
    'lr': [4*0.001, 7*0.001],
    'optimizer__weight_decay': [1e-04, 1e-3],
    'max_epochs': [5],
}

gs = GridSearchCV(clf, param_grid, refit=True, cv=5, verbose=1)
gs.fit(train_X, train_y)
"""
train_set => compatiable for Grid.fit
    check skock with gridsearch
"""

n_epochs = len(gs.best_estimator_.history)
train_loss = np.array([gs.best_estimator_.history[k]['train_loss'] for k in range(len(gs.best_estimator_.history))])
train_accuracy = np.array([gs.best_estimator_.history[k]['train_accuracy'] for k in range(len(gs.best_estimator_.history))])
valid_loss = np.array([gs.best_estimator_.history[k]['valid_loss'] for k in range(len(gs.best_estimator_.history))])
valid_accuracy = np.array([gs.best_estimator_.history[k]['valid_accuracy'] for k in range(len(gs.best_estimator_.history))])

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(range(n_epochs), train_loss, '-o', label='Train')
ax[0].plot(range(n_epochs), valid_loss, '-o', alpha=0.7, label='Valid')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[1].plot(range(n_epochs), train_accuracy, '-o')
ax[1].plot(range(n_epochs), valid_accuracy, '-o', alpha=0.7)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
