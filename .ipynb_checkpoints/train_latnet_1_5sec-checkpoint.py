
import os, sys, glob, pdb
import math, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn # Neural network module
import torch.nn.functional as F # Neural network module
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from eegatt_tools.utils import tensor_info, dl_info
from eegatt_tools.model_utils import CNN1d1_fc1, CNN1d1_fc2, CNN1d2_fc1, CNN1d2_fc2, CNN1d1_h2_fc1, CNN1d1_h2_fc2
from eegatt_tools.train_utils import train_model, initialize_weights, acc_bce_multiout, acc_bce_1out, eval_model, rep_runs, run_multimodel_exps, run_multilr_exps, plot_scores
#from eegatt_tools.sig_vis_utils import plot_multichannel
from eegatt_tools.load_prepare_data_5sec import load_prepare_data_5sec


# GPU vs CPU
if torch.cuda.is_available():
    device = torch.device('cuda:0') # 0 indicates the current device
    print(torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(),
        torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    
device = torch.device('cpu')

writer = SummaryWriter()

train_ds, test_ds, train_filenames, test_filenames = load_prepare_data_5sec(indx_subj=18, indx_subgroup='*g_6')
n_train_samps = len(train_ds)
n_test_samps = len(test_ds)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=n_test_samps, shuffle=True)

dl_info(train_dl)
dl_info(test_dl)

fs=64
n_tps = int(fs*10)
##
# Model

class CNN1d1_fc1(nn.Module):

    def __init__(self, n_inpch=42, n_inpd2=n_tps, n_outs=2, ks1=9, n_ch1=2, n_fc1=2):
        super(CNN1d1_fc1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_inpch, n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(int(n_inpd2 / 2 * n_ch1), n_fc1),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(n_fc1, n_outs)

    def forward(self, x):
        out = self.conv1(x)  # (bs, C, H,  W)
        out = out.view(out.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy
        out = self.fc1(out)
        out = self.fc2(out)
        #import pdb; pdb.set_trace()
        #out = torch.softmax(out, dim=1)  # classes are in columns
        return out


class CNN_francart(nn.Module):

    def __init__(self, n_inpch=42, n_inpd2=n_tps, n_outs=2, ks1=9, n_ch1=5, n_fc1=5):
        super(CNN_francart, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_inpch, n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=n_inpd2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(n_ch1, n_fc1),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(n_fc1, n_outs)

    def forward(self, x):
        out = self.conv1(x)  # (bs, C, H,  W)
        out = out.view(out.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy
        out = self.fc1(out)
        out = self.fc2(out)
        #import pdb; pdb.set_trace()
        #out = torch.softmax(out, dim=1)  # classes are in columns
        return out, 1
    
class EEGLatNet1(nn.Module):
    
    def __init__(self, n_inpch=21, n_inpd2=n_tps, n_outs=2, ks1=9, n_ch1=3, n_ch2=2, n_fc1=2):
        super(EEGLatNet1, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_inpch, n_ch1, kernel_size=ks1, stride= 1, padding=int((ks1-1)/2)),
            nn.ReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_ch1, n_ch2, kernel_size=(2,3), padding=(0, int((ks1-1)/2))),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,n_inpd2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(n_ch2, n_fc1),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(n_fc1, n_outs)
               
    def forward(self, x):
        out = torch.stack((self.conv1(x[:, :21, :]), self.conv1(x[:, 21:42, :])), dim=2)
        #import pdb; pdb.set_trace()
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        #out = self.fc1(out)
        #import pdb; pdb.set_trace()
        out = self.fc2(out)
        return out, 1

    
class EEGLatNet(nn.Module):
    
    def __init__(self, n_inpch=21, n_inpd2=n_tps, n_outs=2, ks1=9, n_ch1=2,  n_fc1=2):
        super(EEGLatNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_inpch, n_ch1, kernel_size=ks1, stride= 1, padding=int((ks1-1)/2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=n_inpd2)
        )
        
        #self.fc1 = nn.Sequential(
         #   nn.Linear(2*n_ch1, n_fc1),
         #   nn.ReLU()
        #)
        self.fc2 = nn.Linear(2*n_ch1, n_outs)
               
    def forward(self, x):
        out = torch.stack((self.conv1(x[:, :21, :]), self.conv1(x[:, 21:42, :])), dim=2)
        #import pdb; pdb.set_trace()
        out = out.view(out.size(0), -1)
        #out = self.fc1(out)
        #import pdb; pdb.set_trace()
        out = self.fc2(out)
        return out, 1    
    
    
    
class CNN_k1(nn.Module):

    def __init__(self, n_inpch=[5, 6, 7, 3, 5, 6, 7, 3], n_inpd2=n_tps, n_outs=2, ks1=9, ks2=3, n_ch1=1, n_fc1=8):
        super(CNN_k1, self).__init__()
        
        self.conv1 = nn.ModuleList([nn.Sequential(nn.Conv1d(n_inpch[j], n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)), nn.ReLU()) for j in range(8)])
        
        self.conv2_1 = nn.ModuleList([nn.Sequential(nn.Conv1d(1, n_ch1, kernel_size=ks2, padding=int((ks2 - 1) / 2)), nn.ReLU(), nn.AvgPool1d(kernel_size=n_inpd2)) for j in range(8)])
        
        self.conv2_2 = nn.ModuleList([nn.Sequential(nn.Conv1d(2, n_ch1, kernel_size=ks2, padding=int((ks2 - 1) / 2)), nn.ReLU(), nn.AvgPool1d(kernel_size=n_inpd2)) for j in range(int(8*7/2))])
   
        self.fc2 = nn.Linear(8+int((8*7)/2), n_outs)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        n_inpch = [5, 6, 7, 3, 5, 6, 7, 3]
        indx_chans = [range(sum(n_inpch[:j]), sum(n_inpch[:j+1])) for j in range(len(n_inpch))]
        out1, out2 = [], []
        for j in range(8):
            out1.append(self.conv1[j](x[:, indx_chans[j],:]))
        out1 = torch.cat(out1, 1)
            
        for j in range(8):
            out2.append(self.conv2_1[j](out1[:, j:j+1, :]))
            
        for j in range(8-1):
            for k in range(j+1, 8):
                out2.append(self.conv2_2[j](out1[:,[j,k],:]))
        
            
        out2 = torch.cat(out2, 1)      

        
        out = out2.view(out2.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy
        out = self.fc2(out)
        #import pdb; pdb.set_trace()
        #out = torch.softmax(out, dim=1)  # classes are in columns
        return out,  out2.mean(dim=0).tolist()
  


torch.manual_seed(123)
model = EEGLatNet1().to(torch.double)
model.to(device)

n_epochs, lr, train_dl, test_dl = 300, 1e-3, train_dl, test_dl
loss_fn, acc_fn, device, verbose = nn.CrossEntropyLoss(), acc_bce_multiout, device, True

loss_batches = []
losses_train = []
accs_test = []
losses_test = []
valid_accs = []
cnt = 0
embedlayer = []

optimizer = optim.Adam(model.parameters(), lr=lr)
# bsize_train = len(train_dl.dataset) # Number of samples in the dataloader

for jep in range(n_epochs):

    for jb, sample_batched in enumerate(train_dl):
        # model.train()
        #import pdb; pdb.set_trace()
        Xb = sample_batched[0].to(device)  # not inplace for tensors
        yb = sample_batched[1].to(device)  # not inplace for tensors
        # import pdb; pdb.set_trace()
        # Forward pass
        optimizer.zero_grad()
        pred, layer8 = model(Xb)
        # import pdb; pdb.set_trace()
        embedlayer.append(layer8)

        loss_batch = loss_fn(pred, yb)  # averaged across samples in the batch
        #writer.add_scalar('Loss/train_batch', loss_batch, cnt)
        loss_batch.backward()
        optimizer.step()

        loss_batches.append(loss_batch.item())
        cnt += 1

    loss_epoch = loss_batches[-(jb + 1):]  # mean loss across batches
    loss_epoch = sum(loss_epoch) / len(loss_epoch)
    writer.add_scalar('Loss/train_epoch', loss_epoch, jep)
    #for jname, jw in model.named_parameters():
    #    if jname == 'fc2.weight':
            #writer.add_histogram(jname,jw, jep)
            #writer.add_histogram(f'{jname}.grad',jw.grad, jep)
    losses_train.append(loss_epoch)  #

    loss_test, acc_test = eval_model(test_dl, model, loss_fn, acc_fn, device=device)
    losses_test.append(loss_test)  # Not a gradient so no item()
    writer.add_scalar('Loss/test_epoch', loss_test, jep)
    writer.add_scalar('ACC/test_epoch', acc_test, jep)
    if verbose:
        print("Epoch: [%2d/%2d], Loss Train: %.4f,    Loss Test: %.4f,  ACC Test:%.2f " % (
        jep + 1, n_epochs, loss_epoch, loss_test, acc_test * 100))


writer.close()

eslayer = torch.stack([torch.tensor(embedlayer[j]) for j in range(len(embedlayer))]).squeeze().cpu().detach().numpy()
plt.plot(eslayer)
#plt.gca().legend(('FL', 'CL', 'PL', 'OL', 'FR', 'CR', 'PR', 'OR'))


#torch.save({
#            'epoch': 160,
#            'model_state_dict': model.state_dict(),
 #           'optimizer_state_dict': optimizer.state_dict(),
 #           'loss': loss_batch
 #           }, f'trainings/sb16')







#for jep in range(n_epochs):

#    for jb, sample_batched in enumerate(train_dl):
        # model.train()
        #import pdb; pdb.set_trace()
#        Xb = sample_batched[0].to(device)  # not inplace for tensors
#        yb = sample_batched[1].to(device)  # not inplace for tensors
        # import pdb; pdb.set_trace()
        # Forward pass
 #       optimizer.zero_grad()
 #       pred, layer8 = model(Xb)
        # import pdb; pdb.set_trace()
#        embedlayer.append(layer8)

 #       loss_batch = loss_fn(pred, yb)  # averaged across samples in the batch
 #       writer.add_scalar('Loss/train_batch', loss_batch, cnt)
  #      loss_batch.backward()
#        optimizer.step()

 #       loss_batches.append(loss_batch.item())
 #       cnt += 1
        
#Xb = sample_batched[0].to(device)  # not inplace for tensors
#yb = sample_batched[1].to(device)  # not inplace for tensors
#m = torch.nn.LogSoftmax()
#pred, layer8 = model(Xb)

#indx_1 = (pred.argmax(dim=1) == yb) & (yb==1)
#indx_0 = (pred.argmax(dim=1) == yb) & (yb==0)


## Add the not-averaged last layer to the model output (mean(averaged-layer * W_end))
## Think about the the last layer, is it the connectivity matrix
## ADD best model training
#https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee


#for name, par in model.named_parameters():
#    print(name)
#    if name == 'fc2.weight':
#        x = par.detach().numpy()
        
#L = x[0, :]
#R = x[1, :]

#Lmat, Rmat = np.zeros((8, 8)), np.zeros((8, 8))
#np.fill_diagonal(Lmat, L[:8])
#np.fill_diagonal(Rmat, R[:8])

#cnt=8
#for j in range(0, 8):
#    for k in range(j+1, 8):
#        Lmat[k, j] = L[cnt]
#        Lmat[j, k] = L[cnt]
#        Rmat[k, j] = R[cnt]
#        Rmat[j, k] = R[cnt]
#        cnt += 1
        
#import seaborn as sns

#labels = ['FL', 'CL', 'PL', 'OL', 'FR', 'CR', 'PR', 'OR']
#sns.heatmap(Lmat, xticklabels=labels, yticklabels=labels, cmap="vlag", center=0)

#sns.heatmap(Rmat, xticklabels=labels, yticklabels=labels,cmap="vlag", center=0)



#hh = eslayer[2249,:]
#hhmat = np.zeros((8,8))
#cnt=8
#for j in range(0, 8):
#    for k in range(j+1, 8):
#        hhmat[k, j] = hh[cnt]
#        hhmat[j, k] = hh[cnt]
#        cnt += 1
        

#import seaborn as sns

#labels = ['FL', 'CL', 'PL', 'OL', 'FR', 'CR', 'PR', 'OR']
#sns.heatmap(hhmat, xticklabels=labels, yticklabels=labels, cmap="flare")