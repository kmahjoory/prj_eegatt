

import os, sys, glob
import math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import pdb
import scipy

from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
#from braindecode.datasets import create_from_X_y

#import torch
#from braindecode.util import set_random_seeds
#from braindecode.models import ShallowFBCSPNet
#from skorch.helper import SliceDataset
#from sklearn.model_selection import train_test_split

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True


# ######################################################################################################################
# PREPARE DATA

subj_id = 10
data_dir = '/hpc/workspace/2021-0292-NeuroFlex/prj_eegatt/data/eeg_after_ica/'

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

montage = mne.channels.make_standard_montage('biosemi64')
eeg_epochs.set_montage(montage)



sfreq = eeg_epochs.info['sfreq']
ch_names = eeg_epochs.info['ch_names']
# Preprocess EEG
eeg_epochs.filter(4, 32)
cov = mne.compute_covariance(eeg_epochs)


## Add adult MRI template (fsaverage)
import os.path as op
import numpy as np

import mne
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation

src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')


src = mne.setup_source_space(subject='fsaverage', spacing='ico4', surface='white', 
                             subjects_dir=subjects_dir, add_dist=True)

mne.bem.make_watershed_bem(subject='fsaverage4', subjects_dir=subjects_dir, overwrite=True)

bem_surfaces = mne.make_bem_model('fsaverage', ico=3, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir)

mne.viz.plot_alignment(
    eeg_epochs.info, src=src, eeg=['projected'], trans=trans,
    show_axes=True, mri_fiducials=True, dig='fiducials')

fwd = mne.make_forward_solution(eeg_epochs.info, trans=trans, src=src, bem=bem, 
                                eeg=True, mindist=5.0, n_jobs=1)
print(fwd)


inv = make_inverse_operator(eeg_epochs.info, fwd, cov, loose=1., depth=0.8,
                            verbose=True)

snr = 3.
lambda2 = 1. / snr ** 2
stc = apply_inverse_epochs(eeg_epochs, inv, lambda2, 'eLORETA', verbose=True)

