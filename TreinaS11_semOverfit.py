import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optm
import numpy as np
import pyTorchTools

device = 'cuda'

def train_Valid(model, patience, n_epochs, scheduler, train_loader,
                      valid_loader, optimizer, criterion, lr_schedul, checkpoint):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the training acc as the model trains
    train_accs = []
    # to track the validation acc as the model trains
    valid_accs = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # to track the average training acc per epoch as the model trains
    avg_train_acc = []
    # to track the average validation acc per epoch as the model trains
    avg_valid_acc = []
    # initialize the early_stopping object
    early_stopping = pyTorchTools.EarlyStopping(patience=patience, verbose=True, name=checkpoint)

    model.to(device)

    # ct = 0
    # for child in model.children():
    #     # print(child)
    #     ct += 1
    #     if ct == 28:  # < 12:
    #         for param in child.parameters():
    #             param.requires_grad = False  # freeze!

    for epoch in range(1, n_epochs + 1):

        # scheduler.step()

        ###################
        # train the model #
        ###################
        lr_schedul.step()
        model.train()  # prep model for training
        for batch, (data, target) in enumerate(train_loader, 1):
            target = target.long()
            data = data.float()
            target = target.to(device)
            # clear the gradients of all optimized variables
            data = data.to(device)
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model.forward(data)
            _, preds = torch.max(output, 1)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            train_accs.append(float(torch.sum(preds == target)) / len(target))

        ######################
        # validate the model #
        ######################
        with torch.no_grad():

            model.eval()  # prep model for evaluation
            for data, target in valid_loader:
                target = target.long()
                data = data.float()
                data = data.to(device)
                target = target.to(device)

                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the loss
                loss = criterion(output, target)
                # record validation loss
                valid_losses.append(loss.item())

                equal = (output.max(dim=1)[1] == target)
                valid_accs.append(float(torch.sum(equal)) / len(target))

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        # # Note that step should be called after validate()
        scheduler.step(valid_loss)

        train_acc = np.average(train_accs)
        valid_acc = np.average(valid_accs)
        avg_train_acc.append(train_acc)
        avg_valid_acc.append(valid_acc)

        epoch_len = len(str(n_epochs))

        if epoch%50 == 0:
            print_msg = "Epoch: {}/{} Train Loss: {:.2f} Valid Loss: {:.5f} Train Acc: {:.5f} Valid Acc: " \
                        "{:.2f}".format(epoch, n_epochs, train_loss, valid_loss, train_acc, valid_acc)

            print(print_msg)

        # clear lists to track next epoch
        preds = []
        equal = []
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []

        # early_stopping(valid_loss, model, checkpoint)
        #
        # if early_stopping.early_stop:
        #     print("Early stopping \n")
        #     break

    # load the last checkpoint with the best model
    # model.load_state_dict(torch.load(checkpoint))

    return model, avg_train_losses, avg_valid_losses, avg_train_acc, avg_valid_acc
