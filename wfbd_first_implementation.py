

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
    window_stride_samples=int(2.5*sfreq),
    window_size_samples=int(5*sfreq),
)
dataset.set_description(events_info)


# ######################################################################################################################
# SEPARATE TEST & TRAIN DATA

indx_test = [10, 15, 21, 29, 49, 55]
split = dataset.split({'train': np.delete(np.arange(60), indx_test), 'test': indx_test})
train_set = split['train']
valid_set = split['test']

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

lr = 0.05 * 0.01
weight_decay = 0.8 * 0.001
batch_size = 64
n_epochs = 20

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
    ],
    device=device,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)


######################################################################
# Visualization

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                  index=clf.history[:, 'epoch'])

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(clf.history[:, 'epoch'], clf.history[:, 'train_loss'], '-o')
ax[0].plot(clf.history[:, 'epoch'], clf.history[:, 'valid_loss'], '-o')
ax[0].set_ylabel('Loss')
ax[0].set_ylim(.2, 2)
ax[1].plot(clf.history[:, 'epoch'], np.array(clf.history[:, 'train_accuracy'])*100, '-o')
ax[1].plot(clf.history[:, 'epoch'], np.array(clf.history[:, 'valid_accuracy'])*100, '-o')
ax[1].set_ylabel('Accuracy')
ax[1].set_ylim(40, 100)

# Plot Confusion Matrix
from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix

y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)
# generating confusion matrix
confusion_mat = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(confusion_mat)
