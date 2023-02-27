import os, sys, glob
import math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import pdb
import scipy


def mk_ml_samples_from_eeg(sample_len_sec, overlap_ratio=0.5):
    """
    This function generates samples for machine learning from EEG data.
    :param sample_len: in seconds
    :param eeg_path: the full-path for the eeg data, including the fif file itself.
    :param events_path: 
    :param write_path: 
    :return: 
    
    Example:
    sample_len_sec=10, overlap_ratio=0.5 for 50% overlap and 10 sec trials
    sample_len_sec=5, overlap_ratio=0 for NO overlap and 5 sec trials
    """


    indx_subjs = np.setdiff1d(range(1, 19), [8, 11])
    write_path = f'../eeg/data_preprocessed/data_eegatt_overlap_{int(overlap_ratio*100)}_{sample_len_sec}_sec_mednorm/train/'
    os.makedirs(f'{write_path}r/', exist_ok=True)
    os.makedirs(f'{write_path}l/', exist_ok=True)

    for jsb in indx_subjs:

        print(jsb)

        eeg_path = f'../eeg/data_preprocessed/eeg_after_ica/subj_{jsb}_eeg_after_ica_epo.fif'
        events_path = f'../eeg/data_preprocessed/eeg_after_ica/subj_{jsb}_expinfo.csv'

        events_info = pd.read_csv(events_path, index_col=0)
        eeg_epochs_ = mne.read_epochs(eeg_path).pick_types(eeg=True)
        eeg_epochs = eeg_epochs_.copy().filter(l_freq=1, h_freq=30).resample(sfreq=64)

        chans_fl = ['Fp1', 'AF3', 'F1', 'F3', 'F5']
        chans_cl = ['FC1', 'FC3', 'FC5', 'C1', 'C3', 'C5']
        chans_pl = ['CP1', 'CP3', 'CP5', 'P1', 'P3', 'P5', 'P7']
        chans_ol = ['PO3', 'PO7', 'O1']

        chans_fr = ['Fp2', 'AF4', 'F2', 'F4', 'F6']
        chans_cr = ['FC2', 'FC4', 'FC6', 'C2', 'C4', 'C6']
        chans_pr = ['CP2', 'CP4', 'CP6', 'P2', 'P4', 'P6', 'P8']
        chans_or = ['PO4', 'PO8', 'O2']

        cnames = [chans_fl, chans_cl, chans_pl, chans_ol, chans_fr, chans_cr, chans_pr, chans_or]
        cnames_label = ['fl', 'cl', 'pl', 'ol', 'fr', 'cr', 'pr', 'or']
        chan_names = [jj for j in cnames for jj in j]
        chan_region = [cnames_label[j] for j in range(len(cnames)) for jj in cnames[j]]

        eeg_epochs = eeg_epochs.pick_channels(ch_names=chan_names, ordered=True)

        assert events_info.shape[0] == eeg_epochs.get_data().shape[
            0]  # Make sure that the loaded info matches with eeg data

        indx_2speakers = events_info['n_speakers'] == 2  # Only 2 speaker conditions are selected
        events_info = events_info[indx_2speakers]
        events_info = events_info.reset_index()  # To reset the index col to start from 0

        fs = eeg_epochs.info['sfreq']  # sampling frequency of EEG after downsampling
        eeg_epochs = eeg_epochs.get_data()
        eeg_epochs = eeg_epochs[indx_2speakers, :, :]  # Pick only 2 speaker conditions

        events_info['attended_story'] = events_info['attended_talker_file'].str.split('story', 1).str.get(1).str.get(0) # Add story information

        n_epochs, n_chans, n_tps = eeg_epochs.shape
        eeg_epochs.reshape([n_epochs * n_tps, n_chans]).shape
        chans_pow = scipy.stats.trim_mean((eeg_epochs ** 2).reshape([n_epochs * n_tps, n_chans]), .1, axis=0)
        med_chans_pow = np.sqrt(np.median(chans_pow))
        eeg_epochs /= med_chans_pow

        print(fs)

        n_epochs, n_chans, ntps_epoch = eeg_epochs.shape
        sample_len = int(sample_len_sec * fs)  # Here we set lenght of samples to 10 sec
        stride = (1-overlap_ratio) * sample_len  #  stride = (1-ratio)*fs
        segs_onset_tp = np.arange(start=0, stop=ntps_epoch-sample_len+1, step=stride).astype('int').tolist()
        n_segs_in_epoch = len(segs_onset_tp)
        segs_in_epochs = np.stack([eeg_epochs[:, :, i:i + sample_len] for i in segs_onset_tp])
        groups = []

        for je in range(n_epochs):  # Iterate over epochs
            jpos = events_info['attend_lr'][je]
            jgender = events_info['attend_mf'][je]
            jac_cond = events_info['acoustic_condition'][je]
            # TODO: ADD audio file
            jstory = events_info['attended_story'][je]

            jg = f'{jgender}{jac_cond}{jstory}'
            if jg not in groups:
                groups.append(jg)         

            jg_indx = groups.index(jg)

            for jseg in range(n_segs_in_epoch):
                # import pdb; pdb.set_trace()
                seg = segs_in_epochs[jseg, je, :, :].squeeze()
                write_name = f'{write_path}/{jpos}/s_{jsb}_e_{je}_seg_{jseg}_gen_{jgender}_ac_{jac_cond}_story_{jstory}_g_{jg_indx}.npy'
                np.save(write_name, seg)
                del seg
            del jpos, jgender, jac_cond, jstory
        del events_info

