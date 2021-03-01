
from collections import OrderedDict

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import itertools

class MLPDropout(nn.Module):
    '''
    A multilayer perception with ReLU activations and dropout layers.
    '''
    def __init__(self, dims, output_dim, dropout_probs):
        '''
        Constructor.
        Parameters
        ----------
        dims : list[int]
            Specifies the input and hidden layer dimensions.
        output_dim : int
            Specifies the output dimension.
        dropout_probs : list[float]
            Specifies the dropout probability at each layer. The length of this
            list must be equal to the length of dims. If the dropout
            probability of a layer is zero, then the dropout layer is omitted
            altogether.
        '''
        if len(dims) != len(dropout_probs):
            raise ValueError('len(dims) must equal len(dropout_probs)')
        if len(dims) < 1:
            raise ValueError('len(dims) must be at least 1')
        if any(prob < 0 or prob > 1 for prob in dropout_probs):
            raise ValueError('Dropout probabilities must be in [0, 1]')

        super(MLPDropout, self).__init__()
        layers = []
        if dropout_probs[0] > 0:  # Input dropout layer.
            layers.append(('Dropout1', nn.Dropout(p=dropout_probs[0])))

        for i in range(len(dims) - 1):
            layers.append((f'Linear{i + 1}', nn.Linear(dims[i], dims[i + 1])))
            layers.append((f'ReLU{i + 1}', nn.ReLU()))
            if dropout_probs[i + 1] > 0:
                dropout = nn.Dropout(p=dropout_probs[i + 1])
                layers.append((f'Dropout{i + 2}', dropout))

        layers.append((f'Linear{len(dims)}', nn.Linear(dims[-1], output_dim)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.model(x)


class MLP(MLPDropout):
    '''
    A multilayer perceptron with ReLU activations.
    '''
    def __init__(self, dims, output_dim):
        '''
        Constructor.
        Parameters
        ----------
        dims : List[int]
            Specifies the input and hidden layer dimensions.
        output_dim : int
            Specifies the output dimension.
        '''
        super(MLP, self).__init__(dims, output_dim, [0] * len(dims))
