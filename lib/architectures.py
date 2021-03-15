"""
    Model architectures.
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_units= 100, num_outputs= 1, use_dropout= False):
        super(Model, self).__init__()
        self.hidden_units = hidden_units
        self.num_outputs  = num_outputs
        self.use_dropout  = use_dropout

        self.net = []
        self.net.append(nn.Linear(in_features= 1, out_features= self.hidden_units))
        if use_dropout:
            self.net.append(nn.Dropout(p = 0.5))
        self.net.append(nn.ReLU())
        # self.net.append(nn.BatchNorm1d(num_features= self.hidden_units))
        # self.net.append(nn.Linear(in_features= self.hidden_units, out_features= self.hidden_units))
        # self.net.append(nn.ReLU())
        # self.net.append(nn.Linear(in_features= self.hidden_units, out_features= self.hidden_units))
        # self.net.append(nn.ReLU())
        self.net = nn.Sequential(*self.net)
        self.out_1 = nn.Linear(in_features= self.hidden_units, out_features= 1)
        self.out_2 = nn.Linear(in_features= self.hidden_units, out_features= 1)

        # Carry out weights initialization
        self.apply(self.init_weights)

        # Finally print the networks
        print(self.net)
        print(self.out_1)
        if self.num_outputs == 2:
            print(self.out_2)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        embeddings = self.net(x)

        mean       = self.out_1(embeddings)
        sigma_sq   = nn.functional.relu(self.out_2(embeddings)) + 1e-3

        if self.num_outputs == 2:
            final_output = torch.cat( (mean, sigma_sq), dim= 1)
        else:
            final_output = mean

        return final_output