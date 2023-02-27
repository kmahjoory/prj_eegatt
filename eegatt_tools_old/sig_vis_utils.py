import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt




def plot_multichannel(X, fs, ax=None, dy_scale=4, title='', spines_bottom=False):
    '''
    Plot the multi-channel data, stacking the channels horizontally on top of each other.

    Parameters
    ----------
    X: numpy array (channels x time points)
        
    dy: float (default 100)
        Amount of vertical space to put between the channels
    color: string (default 'k')
    '''
    if ax is None:
        ax = plt.gca() 
    
    
    dy = dy_scale * np.percentile(X, 75, axis=1).mean() 

    nchannels, ntpoints = X.shape
   
    bases = dy * np.arange(nchannels).reshape((nchannels, 1)) # dy * 0, dy * 1, dy * 2, ..., dy*6
    X = X + bases
    t = np.arange(ntpoints) / fs
    

    ax.plot(t, X.T, linewidth=.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(spines_bottom)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks([])
    if not spines_bottom:
      ax.set_xticks([])
    ax.set_title(title)
    return ax




def plot_psd(X, fs, ax=None, title='', plot=True):
    '''
    Plot power spectrum density using Pwelch method

    Parameters
    ----------
    X: numpy array (channels x time points)

        Amount of vertical space to put between the channels
    color: string (default 'k')
    '''
    import scipy.io as scpio
    from scipy import signal
    
    if ax is None:
        ax = plt.gca() 

    f, P = signal.welch(X, nperseg=fs, noverlap=fs/2, nfft=2*fs, fs=fs)

    indx_xlim_max = np.where(f == 30)[0].item()
    indx_xlim_min = np.where(f == 1)[0].item()
    ax.plot(f[indx_xlim_min:indx_xlim_max], P[:, indx_xlim_min:indx_xlim_max].T, linewidth=.5)
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim([0, 30])
    #ax.set_yticks([])
    ax.set_xticks(range(0, int(30), 5))
    ax.set_title(title)
    return ax



