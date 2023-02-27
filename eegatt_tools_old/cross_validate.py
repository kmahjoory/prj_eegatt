
import torch
import torch.nn as nn
import torch.optim as optim

cos = nn.CosineSimilarity(dim=1)


def kfold_cv(folds):
    """
    folds: a list of folds. len(folds)= number of desired folds
    every element of folds is another list with 2 elements, first one for the train_folds, and the other for the test

    :return:
    """



