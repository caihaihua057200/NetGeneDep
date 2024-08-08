import numpy as np
from torch import nn, optim
from torch.optim import lr_scheduler
import torch
import torchvision
import torch.nn.functional as F


class AttentionMLP(nn.Module):
    def __init__(self):
        super(AttentionMLP, self).__init__()
        self.n1 = 18138
        self.n2 = 50
        self.n3 = 3170
        self.size1 = 50
        self.size2 = 200
        self.encoder1 = nn.Sequential(
            nn.Linear(self.n1, self.size2), nn.BatchNorm1d(self.size2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.encoder1_1 = nn.Sequential(
            nn.Linear(self.size2, self.size1), nn.BatchNorm1d(self.size1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(self.n3, self.size2), nn.BatchNorm1d(self.size2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.encoder3_3 = nn.Sequential(
            nn.Linear(self.size2, self.size1), nn.BatchNorm1d(self.size1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1 + self.size1 * self.size1 * 3, 256), nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64), nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # ,nn.Sigmoid()
        )

        self.Conv = nn.Sequential(
            nn.Conv2d(2, 50, 1), nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.2),
        )

    def forward(self, x, y):
        n = x.shape[1]
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3
        size1 = self.size1
        x1_1 = self.encoder1(x[:, 0:n1])
        x1 = self.encoder1_1(x1_1)
        x2 = x[:, n1:(n1 + n2)]
        x3_3 = self.encoder3(x[:, (n1 + n2):n])
        x3 = self.encoder3_3(x3_3)

        x1_1 = torch.reshape(x1_1, (x1.shape[0], 1, 20, 10))
        x3_3 = torch.reshape(x3_3, (x1.shape[0], 1, 20, 10))
        c = torch.cat((x1_1,x3_3), 1)
        c = self.Conv(c)


        c = torch.reshape(c, (x1.shape[0], size1, size1))
        x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1], 1))
        x2 = torch.reshape(x2, (x2.shape[0], 1, x2.shape[1]))
        x3aaa = torch.bmm(x1, x2)

        x3 = torch.reshape(x3, (x3.shape[0], 1, x3.shape[1]))
        x2aaa = torch.bmm(x1, x3)

        x2 = torch.reshape(x2, (x2.shape[0], x2.shape[2], 1))
        x1aaa = torch.bmm(x2, x3)
        x3a = x1aaa + x2aaa + c
        x2a = x1aaa + x3aaa + c
        x1a = x2aaa + x3aaa + c
        x1a = torch.reshape(x1a, (x1a.shape[0], x1a.shape[1] * x1a.shape[2]))
        x2a = torch.reshape(x2a, (x2a.shape[0], x2a.shape[1] * x2a.shape[2]))
        x3a = torch.reshape(x3a, (x3a.shape[0], x3a.shape[1] * x3a.shape[2]))
        x = torch.cat((x1a, x2a, x3a, y), 1)
        x = self.decoder(x)
        return x
