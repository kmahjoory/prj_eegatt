
import os, sys, time
import torch
import torch.nn as nn
from eegatt_tools.models import normalize_weights_eegnet

def reset_weights(m):
    """
    This function resets the model weights.
    Example:
        model.apply(reset_weights)
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def train_one_epoch(dataloader, model, loss_fn, optimizer, device=torch.device('cpu'), weight_constraint=None, verbose=False):

    loss_batches = []
    hlayer_batches = []

    # Loop over batches in an epoch
    for idx_batch, (Xb, yb) in enumerate(dataloader):

        model.train()

        Xb.to(device)
        yb.to(device)

        # Forward Pass
        pred = model(Xb)

        # Loss computation
        loss_batch = loss_fn(pred, yb)

        # Backward and compute gradients
        optimizer.zero_grad()
        loss_batch.backward()

        # Update Weights/Parameters
        optimizer.step()

        # Apply constraints on weights
        if weight_constraint:
            normalize_weights_eegnet(model)

        loss_batch_value = loss_batch.item()
        loss_batches.append(loss_batch_value)
        #hlayer_batches.append(hlayer)

        # Report batch loss
        if verbose:
            print(f"Batch: {idx_batch},  Loss: {loss_batch_value}")

    return model, loss_batches#, hlayer_batches


def train_model(train_dl, test_dl, model, loss_fn, acc_fn, optimizer, device=torch.device('cpu'), n_epochs=20,
                weight_constraint=False, verbose=True, verbose_batches=False):

    loss_batches = []
    loss_epochs = []
    loss_epochs_test = []
    acc_epochs_test = []
    run_start = time.time()

    best_model_weights = model.state_dict()
    best_acc = 0.0

    for epoch in range(n_epochs):

        # Train one epoch
        model, loss_epoch_batches = train_one_epoch(train_dl, model, loss_fn, optimizer, device,
                                                    weight_constraint=weight_constraint, verbose=verbose_batches)

        loss_epoch = sum(loss_epoch_batches) / len(loss_epoch_batches)
        loss_epochs.append(loss_epoch)
        loss_batches.extend(loss_epoch_batches)

        # Test the model
        test_loss, test_acc = eval_model(test_dl, model, loss_fn, acc_fn, device)
        loss_epochs_test.append(test_loss)
        acc_epochs_test.append(test_acc)

        # Save the best model
        if not os.path.exists('best_models'):
            os.mkdir('best_models')

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_weights = model.state_dict().copy()
            torch.save(best_model_weights, f"./best_models/epoch_{epoch}_acc_{best_acc:.2f}.pth")

        print(f"[Epoch {epoch+1}]: Train Loss: {loss_epoch:0.3f}    Test Loss: {test_loss:0.3f}, "
              f"Test Acc: {test_acc*100:0.2f}%")

    time_elapsed = time.time() - run_start
    print("========================================================")
    print(f'Training of {n_epochs} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.')
    print("========================================================")
    print(f'Best val Acc: {best_acc:.2f}')
    print("========================================================")

    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model, {"loss_batches": loss_batches, "loss_train": loss_epochs, "loss_test": loss_epochs_test,
                   "acc_test": acc_epochs_test, "best_acc": best_acc}


def eval_model(test_dl, model, loss_fn, acc_fn, device=torch.device('cpu')):
    """ This function evaluates performance of a trained model on test data
    Parameters:
    Returns:
    loss_test: Mean loss of all test samples (scalar)
    acc_test: Accuracy of the model on all test samples
    """
    losses, accs = [], []
    model.to(device)  # inplace for model

    # Switch off backward gradient computation
    model.eval()

    loss_batch, acc_batch = 0, 0

    with torch.no_grad():
        for idx_batch, (Xb, yb) in enumerate(test_dl):

            Xb = Xb.to(device)
            yb = yb.to(device)

            # Forward pass
            pred = model(Xb)

            if pred.dim() == 1:
                loss_batch = loss_fn(pred.data, yb.data).item()
                acc_batch = ((pred >= 0.5) == yb).type(torch.float).sum().item()
            else:
                # TO DO: loss and acc function for single neuron output
                if pred.shape[1] == 1:
                    loss_batch = loss_fn(pred.data, yb.data.type(torch.float).view(-1, 1)).item()  # pred.data contains no gradient
                    acc_batch = (pred.argmax(dim=1) == yb).type(torch.float).sum().item()  # accuracy function
                    # This corresponds to two neuron output and nn.CrossEntropyLoss().
                    # In this case both prediction and y should be 1) batchsize x 1 dimenstion. 2) float
                elif pred.shape[1] >= 2:  # Only for cases with 2 or more classes
                    loss_batch = loss_fn(pred.data, yb.data).item()  # pred.data contains no gradient
                    acc_batch = acc_fn(pred, yb)

            batch_size = yb.shape[0]
            losses.append(loss_batch) # Here no need for division. Loss function gives mean loss across samples in the batch
            accs.append(acc_batch / batch_size)

    loss_test = sum(losses) / len(losses) # Loss and accuracy are computed from mean of all test samples. For train set, we used only loss of last bach in epoch!
    acc_test = sum(accs) / len(accs)

    return loss_test, acc_test

def acc_bce_multiout(pred, y):
  # Accuracy for multi-output cross entropy loss
  # This corresponds to two neuron output and nn.CrossEntropyLoss().
  # In this case both prediction and y should be 1) batchsize x 1 dimenstion. 2) float
  #import pdb; pdb.set_trace()
  correct = (pred.argmax(dim=1) == y).type(torch.float).sum().item()  # accuracy function
  return correct


def acc_bce_1out(pred, y):
  # Accuracy for 1 output binary cross entropy loss
  correct = ((pred >= 0.5) == y.view((-1, 1))).type(torch.float).sum().item()
  # pred is computed for all batches, and batches are in dimension 0. So we always use a column vector for  y
  return correct
