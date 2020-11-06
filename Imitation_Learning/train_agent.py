from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import Model, stateDataSet
from utils import *


import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data_2.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"])
    y = np.array(data["action"])

    # get rid of opening picture
    # terminate_idx = np.array(result['terminal'])
    # discard_array = np.array([0]*terminate_idx.shape[0])

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    
    # switch to grayscale 
    # X_train_flat, X_valid_flat = data_preprocess(X_train), data_preprocess(X_valid)
    X_train = np.array([data_preprocess(X_train[i]) for i in range(X_train.shape[0])]).astype('float32') # n x 1 x 84 x 84
    X_valid = np.array([data_preprocess(X_valid[i]) for i in range(X_valid.shape[0])]).astype('float32')
    # stack historical data into channels
    # X_train = np.array([np.stack((X_train_flat[i-history_length:i]), axis=-1) for i in range(history_length, X_train_flat.shape[0]+1)])
    # X_valid = np.array([np.stack((X_valid_flat[i-history_length:i]), axis=-1) for i in range(history_length, X_valid_flat.shape[0]+1)])
    # discretization
    y_train = np.apply_along_axis(action_to_id, 1, y_train)
    y_valid = np.apply_along_axis(action_to_id, 1, y_valid)
    # y_train = y_train[history_length-1:]
    # y_valid = y_valid[history_length-1:]
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")

    history_length = 1
    # specify your neural network in model.py 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    agent = Model(history_length, device=device)
    agent.to(device)
    writer = SummaryWriter()
    
    X_train = torch.tensor(X_train, requires_grad=True)
    y_train = torch.tensor(y_train,).long()
    X_valid = torch.tensor(X_valid,)
    y_valid = torch.tensor(y_valid, device=device).long()
    train_ds = stateDataSet(X_train, y_train,)

    train_data_loader = DataLoader(train_ds, batch_size=batch_size,)

    criterion = CrossEntropyLoss()
    # n_epochs = 200
    optimizer = torch.optim.SGD(agent.parameters(), lr = lr, momentum=0.7)
    
    all_train_loss = []
    all_train_acc = []
    all_valid_loss = []
    all_vliad_acc = []
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0 
    for idx, (batch, target) in enumerate(train_data_loader):
        target = target.to(device)
        agent.train()
        optimizer.zero_grad()
        y_pred = agent(batch)
        _, y_pred_lab = torch.max(y_pred, 1)

        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()

        print('-'*5, 'train ', idx, '-'*5)
        if idx % 20 == 0:
            agent.save(os.path.join(model_dir, "agent_"+str(idx)+".pt"))
        train_loss = loss.item()
        all_train_loss.append(train_loss)
        train_acc = torch.sum(y_pred_lab == target).float()/target.shape[0]
        all_train_acc.append(train_acc)

        agent.eval()
        y_pred = agent(X_valid)
        _, y_pred_lab = torch.max(y_pred, 1)
        loss = criterion(y_pred, y_valid)
        val_loss = loss.item()
        all_valid_loss.append(val_loss)
        val_acc = torch.sum(y_pred_lab==y_valid).float()/y_valid.shape[0]
        all_vliad_acc.append(val_acc)

        print(f'training loss: {train_loss}')
        print(f'validation loss: {val_loss}')
        print(f'training accuracy: {train_acc}')
        print(f'validation accuracy: {val_acc}')
        writer.add_scalars("training/validation", {'loss/training':train_loss, 'loss/validation':val_loss, 'acc/training':train_acc, 'acc/validation':val_acc}, idx)

      
    # save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)
    writer.add_graph(agent, batch)

    writer.close()


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid,)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=1000, batch_size=256, lr=0.001)
 
