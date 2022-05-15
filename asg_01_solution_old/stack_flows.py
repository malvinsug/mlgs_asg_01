from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

import nf_utils as nf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

class StackedFlows(nn.Module):
    """Stack a list of tranformations with a given based distribtuion.

    Args:
        tranforms: list fo stacked tranformations. list of Flows
        dim: dimension of input/output data. int
        base_dist: name of the base distribution. options: ['Normal']
    """
    def __init__(
        self, 
        transforms: List[nf.Flow], 
        dim: int = 2, 
        base_dist: str = 'Normal'
    ):
        super().__init__()
        
        if isinstance(transforms, nf.Flow):
            self.transforms = nn.ModuleList([transforms, ])
        elif isinstance(transforms, list):
            if not all(isinstance(t, nf.Flow) for t in transforms):
                raise ValueError("transforms must be a Flow or a list of Flows")
            self.transforms = nn.ModuleList(transforms)
        else:
            raise ValueError(f"transforms must a Flow or a list, but was {type(transforms)}")
            
        self.dim = dim
        if base_dist == "Normal":
            self.base_dist = MultivariateNormal(torch.zeros(self.dim).to(device), torch.eye(self.dim).to(device))
        else:
            raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of a batch of data (slide 27).

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            log_prob: Log probability of the data, shape [batch_size]
        """
        
        B, D = x.shape

        ##########################################################
        # YOUR CODE HERE
        
        ##########################################################
        
        assert log_prob.shape == (B,)

        return log_prob

    def rsample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample from the transformed distribution (slide 31).

        Returns:
            x: sample after forward tranformation, shape [batch_size, dim]
            log_prob: Log probability of x, shape [batch_size]
        """
        ##########################################################
        # YOUR CODE HERE
        
        ##########################################################

        assert x.shape == (batch_size, self.dim)
        assert log_prob.shape == (batch_size,)

        return x, log_prob