
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
from eegatt_tools.load_prepare_data_1 import load_prepare_data_1


# GPU vs CPU
if torch.cuda.is_available():
    device = torch.device('cuda:0') # 0 indicates the current device
    print(torch.cuda.is_available(), torch.cuda.current_device(), torch.cuda.device_count(),
        torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    
device = torch.device('cpu')

writer = SummaryWriter()

train_ds, test_ds = load_prepare_data_1(indx_subj=18)
n_train_samps = len(train_ds)
n_test_samps = len(test_ds)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=n_test_samps, shuffle=True)

dl_info(train_dl)
dl_info(test_dl)


##
# Model

class CNN1d1_fc1(nn.Module):

    def __init__(self, n_inpch=42, n_inpd2=640, n_outs=2, ks1=9, n_ch1=2, n_fc1=2):
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

    def __init__(self, n_inpch=42, n_inpd2=640, n_outs=2, ks1=9, n_ch1=5, n_fc1=5):
        super(CNN_francart, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_inpch, n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=640)
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
    

    
class CNN_k1(nn.Module):

    def __init__(self, n_inpch=[5, 6, 7, 3, 5, 6, 7, 3], n_inpd2=640, n_outs=2, ks1=9, n_ch1=1, n_fc1=8):
        super(CNN_k1, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_inpch[0], n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=640)
        )
        
                
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_inpch[1], n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=640)
        )
        
                
        self.conv3 = nn.Sequential(
            nn.Conv1d(n_inpch[2], n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=640)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(n_inpch[3], n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=640)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(n_inpch[4], n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=640)
        )
        
                
        self.conv6 = nn.Sequential(
            nn.Conv1d(n_inpch[5], n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=640)
        )
        
                
        self.conv7 = nn.Sequential(
            nn.Conv1d(n_inpch[6], n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=640)
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv1d(n_inpch[7], n_ch1, kernel_size=ks1, padding=int((ks1 - 1) / 2)),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=640)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(8*n_ch1, n_fc1),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(n_fc1, n_outs)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        out1 = torch.cat((self.conv1(x[:,:5,:]), self.conv2(x[:,5:11,:])), 1)  # (bs, C, H,  W)      
        out2 = torch.cat((self.conv3(x[:,11:18,:]), self.conv4(x[:,18:21,:])), 1)  # (bs, C, H,  W)
        out3 = torch.cat((out1, out2), 1)
                          
        out4 = torch.cat((self.conv5(x[:,21:26,:]), self.conv6(x[:,26:32,:])), 1)  # (bs, C, H,  W)
        out5 = torch.cat((self.conv7(x[:,32:39,:]), self.conv8(x[:,39:42,:])), 1)  # (bs, C, H,  W)
        out6 = torch.cat((out4, out5), 1)  
                          
        out7 = torch.cat((out3, out6), 1)
        #import pdb; pdb.set_trace()
        out = out7.view(out7.size(0), -1)  # (bs, C * H, W), # view is like reshape in numpy
        out = self.fc1(out)
        out = self.fc2(out)
        #import pdb; pdb.set_trace()
        #out = torch.softmax(out, dim=1)  # classes are in columns
        return out,  out7.mean(dim=0).tolist()
    


model = CNN_k1().to(torch.double)
model.to(device)

n_epochs, lr, train_dl, test_dl = 100, 1e-3, train_dl, test_dl
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
        writer.add_scalar('Loss/train_batch', loss_batch, cnt)
        loss_batch.backward()
        optimizer.step()

        loss_batches.append(loss_batch.item())
        cnt += 1

    loss_epoch = loss_batches[-(jb + 1):]  # mean loss across batches
    loss_epoch = sum(loss_epoch) / len(loss_epoch)
    writer.add_scalar('Loss/train_epoch', loss_epoch, jep)
    for jname, jw in model.named_parameters():
        writer.add_histogram(jname,jw, jep)
        writer.add_histogram(f'{jname}.grad',jw.grad, jep)
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
plt.gca().legend(('FL', 'CL', 'PL', 'OL', 'FR', 'CR', 'PR', 'OR'))