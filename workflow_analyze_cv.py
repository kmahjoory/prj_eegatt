import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


read_path = '/hpc/workspace/2021-0292-NeuroFlex/prj_eegatt/analyses/cv_20220503_113532/'
read_file = 'mean_accuracy.csv'

data = pd.read_csv(os.path.join(read_path, read_file))
print(data)

lrs = sorted(data['learning_rate'].unique())
wds = sorted(data['weight_decay'].unique())


for wd in wds:
    dat_ = data[data['weight_decay']==wd]
    x = np.arange(dat_['learning_rate'].shape[0])
    plt.plot(x, dat_['acc_mean'], label=wd)
    plt.xticks(x, dat_['learning_rate'])
    plt.legend()
    plt.xlabel('lr')
    plt.ylabel('loss')



for lr_best in lrs:
    for wd_best in wds:

        metrics_fname = f'metrics_fold_lr_{lr_best}_wd_{wd_best}.json'
        read_dir = os.path.join(read_path, metrics_fname)
        with open(read_dir, 'r') as f:
            metrics = json.load(f)


        fig, ax = plt.subplots(1, 5, sharey=True, figsize=(14, 6))
        fig.suptitle(f"lr={lr_best}, wd={wd_best}", fontsize=14)
        for idx, fold in enumerate(metrics):
            ax[idx].plot(fold['loss_train'])
            ax[idx].plot(fold['loss_test'], alpha=0.7)
            ax[idx].set_ylim(0.64, .72)
            ax[idx].set_title(f"Fold {idx+1}, ACC={fold['best_acc']*100+9:0.1f}%")