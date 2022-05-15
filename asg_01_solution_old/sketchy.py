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

# def calculate_jacobian(y, x, retain_graph=False):
#     x_grads = []
#     for xi, yi in enumerate(y.flatten()):
#         if x.grad is not None:
#             x.grad.zero_()
#         # if specified set retain_graph=False on last iteration to clean up
#         
#         yi.backward(retain_graph=retain_graph or xi < y.numel() - 1)
#         x_grads.append(x.grad.clone()) # this one should be flattening the clone and sum it up.
#     return torch.stack(x_grads).reshape(*y.shape, *x.shape)

#def inversed_affine_transform_jacobian_elementwise()

#def inversed_affine_transform_jacobian(inversed_affine,variable):
    
def affine_transform(x):
    return torch.exp(log_scale) * x + shift

def inversed_affine_transform(yi):
    yi_bi_diff = (yi-shift)
    inversed_exp_a = 1/torch.exp(log_scale)
    return yi_bi_diff*inversed_exp_a

# scalar vector
log_scale = nn.Parameter(torch.zeros(2))
shift = nn.Parameter(torch.zeros(2))

data_wrapper = nf.CircleGaussiansDataset(n_gaussians=1,n_samples=3,radius=3.0,variance=0.3, seed=45)
x_data = torch.tensor(data_wrapper.X,requires_grad=True)
x_toy_data = torch.tensor([[1,2],[3,4],[5,6],[7,8]],requires_grad = True)

# Sample z_0 ~ p_0(z_0)
mu,sigma = torch.mean(x_data,axis=0),torch.std(x_data,axis=0)

torch.apply_

f_z = affine_transform(x_data)
f_z_leaf = f_z.detach()
inversed_f_z = inversed_affine_transform(f_z)
variable_x = x_data

def affine_transform(x):


def compute_affine_determinant(f):
    pass

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

    




#torch_inversed_transform_a = torch.inverse(transform_a)
#jacobian_inversed_transform_a = calculate_jacobian(inversed_transform_a,transform_a)
# np.random.normal(mu, sigma, 1000)
# Compute Tranformation
#variable_x = torch.autograd.Variable(x_data,requires_grad=True)





#f_transform.backward()
#f_derivative = f_transform.backward()
#a = torch.tensor([1., 2., 3.])
#b = torch.tensor([4., 5., 6.], requires_grad=True)
#c = a * b
#jacobian = calculate_jacobian(c, b) # dc/db




