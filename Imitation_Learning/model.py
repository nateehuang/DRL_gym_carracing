# import tensorflow as tf
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, history_length = 1, device = 'cpu'):
        super().__init__()
        self.device = device
        self.cov = nn.Sequential(
            nn.Conv2d(history_length, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
        )
        self.batch1 = nn.BatchNorm1d(64*7*7)
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Linear(128, 5)
        )


    def forward(self, x):
        x = x.to(self.device)

        x = self.cov(x)

        x = x.view(x.size()[0], -1)
        x = self.batch1(x)
        x = F.dropout(x)

        x = self.fc(x)

        return x


    def load(self, file_name):

        self.load_state_dict(torch.load(file_name))
        print(f'{file_name} is loaded')
        return self

    def save(self, file_name):

        torch.save(self.state_dict(), f=file_name)
        print(f'{file_name} is saved')

class stateDataSet(Dataset):
    def __init__(self, state_data, action):
        self.state = state_data
        self.action = action
    
    def __len__(self): return self.action.shape[0]

    def __getitem__(self, idx):
        return [self.state[idx], self.action[idx]]
