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

class Affine(nf.Flow):
    """Affine transformation y = e^a * x + b.
    
    Args:
        dim (int): dimension of input/output data. int
    """
    def __init__(self, dim: int = 2):
        """ Create and init an affine transformation. """
        super().__init__()
        self.dim = dim
        self.log_scale = nn.Parameter(torch.zeros(self.dim))  # a
        self.shift = nn.Parameter(torch.zeros(self.dim))  # b

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the forward transformation given an input x.

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            y: sample after forward tranformation. shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward tranformation, shape [batch_size]
        """
        B, D = x.shape
        
        ##########################################################
        # YOUR CODE HERE
        def affine_transform(data_x):
            return torch.exp(self.log_scale) * data_x + self.shift
        
        def compute_affine_jacobian(f):
            dim = f.shape[1]
            identity_matrix = torch.eye(dim)
            exp_log_scale = torch.exp(log_scale)
            return torch.multiply(identity_matrix,exp_log_scale)

        def calculate_determinant(affine_jacobian):
            row_size,column_size = affine_jacobian.shape[0],affine_jacobian.shape[1]
            assert  row_size == column_size
            determinant = 1
            for i in range(row_size):
                determinant *= affine_jacobian[i][i]
            return determinant
        
        for i in range(B):
            f_z = affine_transform(x[i].T)
            jacobian_f_z = compute_affine_jacobian(f_z)
            determinant_f_z = calculate_determinant(jacobian_f_z)
            log_det_jac = torch.log(determinant_f_z)


        ##########################################################
        
        assert y.shape == (B, D)
        assert log_det_jac.shape == (B,)

        return y, log_det_jac

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse tranformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse tranformation, shape [batch_size]
        """
        B, D = y.shape

        ##########################################################
        # YOUR CODE HERE

        ##########################################################

        assert x.shape == (B, D)
        assert inv_log_det_jac.shape == (B,)

        return x, inv_log_det_jac