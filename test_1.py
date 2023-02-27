
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



subj_id = 1

#data_dir = '../data/eeg_preprocessed/'
data_dir = '/hpc/workspace/2021-0292-NeuroFlex/prj_eegatt/data/eeg_preprocessed/'
subj_eeg_name = f'subj_{subj_id}_eeg_after_ica.fif'
subj_event_name = f'subj_{subj_id}_expinfo.csv'
eeg_file_path = os.path.join(data_dir, subj_eeg_name)
event_file_path = os.path.join(data_dir, subj_event_name)
events_info = pd.read_csv(event_file_path, index_col=0)
indx_2speakers = events_info['n_speakers'] == 2
events_info = events_info[indx_2speakers].reset_index(drop=True)

eeg_epochs = mne.read_epochs(eeg_file_path).pick_types(eeg=True)[indx_2speakers]
eeg_epochs.filter(1, 32)

eeg = eeg_epochs.get_data().copy()
sfreq = eeg_epochs.info['sfreq']
ch_names = eeg_epochs.info['ch_names']



X = eeg * 1e3
y = (events_info['attend_lr']=='r').astype('int').tolist()

# Convert to data format compatible with skorch and braindecode:
windows_dataset = create_from_X_y(
    X, y,
    drop_last_window=False,
    sfreq=sfreq,
    ch_names=ch_names,
    window_stride_samples=int(2.5*sfreq),
    window_size_samples=int(5*sfreq),
)
windows_dataset.set_description(events_info)










indx_test = [10, 15, 21, 29, 49, 55]
split2 = windows_dataset.split({'train': np.delete(np.arange(60), indx_test), 'test': indx_test})
train_set = split2['train']
valid_set = split2['test']

########################################################################################################################
## CREATE MODEL

seed = 20200220
#set_random_seeds(seed=seed)

n_classes = 2
# Extract number of chans and time steps from dataset
n_chans = len(ch_names)
input_window_samples = train_set[0][0].shape[1]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)




from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
# These values we found good for shallow network:
lr = 0.05 * 0.01#0.5 * 0.001
weight_decay = 1 * 0.001

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

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
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)




######################################################################
# Now we use the history stored by Skorch throughout training to plot
# accuracy and loss curves.
#

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                  index=clf.history[:, 'epoch'])

# get percent of misclass for better visual comparison to loss
df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
               valid_misclass=100 - 100 * df.valid_accuracy)

plt.style.use('seaborn')
fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(
    ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['train_misclass', 'valid_misclass']].plot(
    ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
ax1.set_xlabel("Epoch", fontsize=14)



# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()


######################################################################
# Plot Confusion Matrix
# ---------------------
#


#######################################################################
# Generate a confusion matrix as in https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
#


from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix

# generate confusion matrices
# get the targets
y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)

# generating confusion matrix
confusion_mat = confusion_matrix(y_true, y_pred)

# add class labels
# label_dict is class_name : str -> i_class : int
label_dict = valid_set.datasets[0].windows.event_id.items()
# sort the labels by values (values are integer class labels)
labels = list(dict(sorted(list(label_dict), key=lambda kv: kv[1])).keys())

# plot the basic conf. matrix
plot_confusion_matrix(confusion_mat)
