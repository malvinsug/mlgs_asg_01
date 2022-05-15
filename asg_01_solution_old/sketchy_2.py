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

def affine_transform(data_x):
    return torch.exp(log_scale) * data_x + shift

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

# scalar vector
log_scale = nn.Parameter(torch.zeros(2))
shift = nn.Parameter(torch.zeros(2))

data_wrapper = nf.CircleGaussiansDataset(n_gaussians=1,n_samples=3,radius=3.0,variance=0.3, seed=45)
x = torch.tensor(data_wrapper.X,requires_grad=True)

B,D = x.shape[0],x.shape[1]

for i in range(B):
    f_z = affine_transform(x[i].T)
    jacobian_f_z = compute_affine_jacobian(f_z)
    determinant_f_z = calculate_determinant(jacobian_f_z)
    log_det_jac = torch.log(determinant_f_z)

