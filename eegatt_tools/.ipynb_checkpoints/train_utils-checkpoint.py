import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor



###########################
## Training & Evaluation
###########################
def train_model(pars):

  n_epochs, lr, train_dl, test_dl, model = pars['n_epochs'], pars['lr'], pars['train_dl'], pars['test_dl'], pars['model']
  loss_fn, acc_fn, device, verbose = pars['loss_fn'], pars['acc_fn'], pars['device'], pars['verbose']

  loss_batches = []
  losses_train = []
  accs_test = []
  losses_test = []
  valid_accs = []
  model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  #bsize_train = len(train_dl.dataset) # Number of samples in the dataloader

  for jep in range(n_epochs):
    
    for jb, sample_batched in enumerate(train_dl):
      
      #model.train()
      #import pdb; pdb.set_trace()
      Xb = sample_batched[0].to(device) # not inplance for tensors
      
      yb = sample_batched[1].to(device) # not inplace for tensors
      #import pdb; pdb.set_trace()
      # Forward pass
      optimizer.zero_grad()
      pred = model(Xb)
      #import pdb; pdb.set_trace()

      loss_batch = loss_fn(pred, yb) # averaged across samples in the batch
      loss_batch.backward()
      optimizer.step()

      loss_batches.append(loss_batch.item())

    loss_epoch = loss_batches[-(jb+1):]  # mean loss across batches
    loss_epoch = sum(loss_epoch) / len(loss_epoch)
    losses_train.append(loss_epoch) # 


    loss_test, acc_test = eval_model(test_dl, model, loss_fn, acc_fn, device=device)
    losses_test.append(loss_test) # Not a gradient so no item()
    if verbose:
      print("Epoch: [%2d/%2d], Loss Train: %.4f,    Loss Test: %.4f,  ACC Test:%.2f " %(jep+1, n_epochs, loss_epoch, loss_test, acc_test*100))
  return losses_train, losses_test, acc_test*100





def train_model_scheduler(pars):

  n_epochs, lr, train_dl, test_dl, model = pars['n_epochs'], pars['lr'], pars['train_dl'], pars['test_dl'], pars['model']
  loss_fn, acc_fn, device, verbose = pars['loss_fn'], pars['acc_fn'], pars['device'], pars['verbose']

  loss_batches = []
  losses_train = []
  accs_test = []
  losses_test = []
  valid_accs = []
  model.to(device)
  optimizer = optim.Adam(model.parameters(), lr = lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  #bsize_train = len(train_dl.dataset) # Number of samples in the dataloader

  for jep in range(n_epochs):
    
    for jb, sample_batched in enumerate(train_dl):
      
      #model.train()
      #import pdb; pdb.set_trace()
      Xb = sample_batched[0].to(device) # not inplance for tensors
      
      yb = sample_batched[1].to(device) # not inplace for tensors
      #import pdb; pdb.set_trace()
      # Forward pass
      optimizer.zero_grad()
      pred = model(Xb)
      #import pdb; pdb.set_trace()

      loss_batch = loss_fn(pred, yb) # averaged across samples in the batch
      loss_batch.backward()
      optimizer.step()

      loss_batches.append(loss_batch.item())

    loss_epoch = loss_batches[-(jb+1):]  # mean loss across batches
    loss_epoch = sum(loss_epoch) / len(loss_epoch)
    losses_train.append(loss_epoch) # 


    loss_test, acc_test = eval_model(test_dl, model, loss_fn, acc_fn, device=device)
    scheduler.step(loss_test)
    losses_test.append(loss_test) # Not a gradient so no item()
    if verbose:
      print("Epoch: [%2d/%2d], Loss Train: %.4f,    Loss Test: %.4f,  ACC Test:%.2f " %(jep+1, n_epochs, loss_epoch, loss_test, acc_test*100))
  return losses_train, losses_test, acc_test*100


def initialize_weights(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
      torch.nn.init.xavier_uniform_(m.weight.data)
      if m.bias is not None:
        torch.nn.init.zeros_(m.bias)
  if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)


def acc_bce_multiout(pred, y):
  # Accuracy for multi-output cross entropy loss
  # This corresponds to two neuron output and nn.CrossEntropyLoss(). 
  # In this case both prediction and y should be 1) batchsize x 1 dimenstion. 2) float
  #import pdb; pdb.set_trace()
  correct = (pred.argmax(dim=1) == y).type(torch.float).sum().item() # accuracy function
  return correct


def acc_bce_1out(pred, y):
  # Accuracy for 1 output binary cross entropy loss
  correct = ((pred >= 0.5) == y.view((-1, 1))).type(torch.float).sum().item()
  # pred is computed for all batches, and batches are in dimension 0. So we always use a column vector for  y
  return correct


def eval_model(valid_dl, model, loss_fn, acc_fn, device):

  """ This function evaluates performance of a trained model on test data

  Parameters:

  Returns:
  loss_test: Mean loss of all test samples (scalar)
  acc_test: Accuracy of the model on all test samples

  """

  losses, accs = [], []
  #model.to(device) # inplace for model
  model.eval()
  loss_batch, acc_batch = 0, 0

  with torch.no_grad():
    for sample_test in valid_dl:
      #import pdb; pdb.set_trace()
      X_test = sample_test[0].to(device)
      y_test = sample_test[1].to(device)
      pred_test,_ = model(X_test)
      #import pdb; pdb.set_trace()
      # TO DO: loss and acc function for single neuron output
      if pred_test.shape[1]==1:  
        loss_batch = loss_fn(pred_test.data, y_test.data.type(torch.float).view(-1, 1)).item() # pred.data contains no gradient
        acc_batch = (pred_test.argmax(dim=1) == y_test).type(torch.float).sum().item() # accuracy function
      # This corresponds to two neuron output and nn.CrossEntropyLoss(). 
      # In this case both prediction and y should be 1) batchsize x 1 dimenstion. 2) float
      elif pred_test.shape[1]>=2: # Only for cases with 2 or more classes 
        loss_batch = loss_fn(pred_test.data, y_test.data).item() # pred.data contains no gradient
        acc_batch = acc_fn(pred_test, y_test)

      batch_size = y_test.shape[0]
      losses.append(loss_batch) # Here no need for division. Loss function gives mean loss across samples in the batch
      accs.append(acc_batch/batch_size)

    loss_test = sum(losses)/len(losses) # Loss and accuracy are computed from mean of all test samples. For train set, we used only loss of last bach in epoch!
    acc_test = sum(accs)/ len(accs)

  return loss_test, acc_test

###########################
## MULTI RUN Analysis
###########################
def rep_runs(pars, scheduler=True):
  """
  This function repeats runs with initializing weights.
  """
  n_reps = pars['n_reps']
  model = pars['model']
  losses_train_exp = []
  losses_test_exp = []
  accs_exp = []
  for jrun in range(n_reps):
    print(f'RUN {jrun+1}')
    model.apply(initialize_weights)
    pars['model'] = model
    if scheduler:
      losses_train, losses_test, acc = train_model_scheduler(pars)
    else:
      losses_train, losses_test, acc = train_model(pars)

    losses_train_exp.append(losses_train)
    losses_test_exp.append(losses_test)
    accs_exp.append(acc)
    scores = {'losses_train':np.array(losses_train_exp), 'losses_test':np.array(losses_test_exp), 'accs':accs_exp}
  return scores


def run_multimodel_exps(models_set, pars, scheduler=True):
  """
  This function runs training on every model several times
  """
  models = models_set[0]
  descs = models_set[1]
  scores_exps = []
  for jexp, p in enumerate(models):
    print(f'EXP # {jexp+1}: {descs["hps"][jexp]}')
    pars['model'] = p
    scores = rep_runs(pars, scheduler=scheduler)
    scores_exps.append(scores)

  return scores_exps, descs



def run_multilr_exps(lr_set, pars, scheduler=True):
  """
  This function runs training on every model several times
  """

  lrs = lr_set
  descs = {'hps': [str(jlr) for jlr in lr_set], 'write_name': [str(jlr) for jlr in lr_set]}
  n_reps = pars['n_reps']
  scores_exps = []
  for jexp, p in enumerate(lrs):
    print(f'EXP # {jexp+1}: {descs["hps"][jexp]}')
    pars['lr'] = p
    scores = rep_runs(pars)
    scores_exps.append(scores)

  return scores_exps, descs


def plot_scores(scores_exps):
  write_path = '/content/drive/My Drive/Colab Notebooks/tutorials/analysis_outs/'
  n_exps = len(scores_exps[0])
  accs_exps = [scores_exps[0][j]['accs'] for j in range(n_exps)]
  write_name = scores_exps[1]['write_name']
  labels = scores_exps[1]['hps']
  losses_train_exps = [np.array(scores_exps[0][j]['losses_train']) for j in range(n_exps) ]
  losses_test_exps = [np.array(scores_exps[0][j]['losses_test']) for j in range(n_exps) ]
  n_cols = n_exps if n_exps!=1 else n_exps+1
  fig, axs = plt.subplots(3,n_cols, sharey='row', figsize=(5*n_exps,10))
  for jexp in range(n_exps):
    axs[0, jexp].boxplot(accs_exps[jexp])
    axs[1,jexp].plot(losses_train_exps[jexp].T)
    axs[2,jexp].plot(losses_test_exps[jexp].T)
    axs[0, jexp].set_title(labels[jexp])
    axs[0, jexp].yaxis.grid(True)
  plt.subplots_adjust(hspace=0, wspace=0 )
  plt.savefig(f"{write_path}{write_name}.jpg", bbox_inches='tight')



  


