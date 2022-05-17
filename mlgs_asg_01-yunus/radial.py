from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

import nf_utils as nf

class Radial(nf.Flow):
    """Radial transformation.

    Args:
        dim: dimension of input/output data, int
    """

    def __init__(self, dim: int = 2):
        """ Create and initialize an affine transformation. """
        super().__init__()

        self.dim = dim

        self.x0 = nn.Parameter(
            torch.Tensor(self.dim, ))  # Vector used to parametrize z_0
        self.pre_alpha = nn.Parameter(
            torch.Tensor(1, ))  # Scalar used to indirectly parametrized \alpha
        self.pre_beta = nn.Parameter(
            torch.Tensor(1, ))  # Scaler used to indireclty parametrized \beta

        stdv = 1. / np.sqrt(self.dim)
        self.pre_alpha.data.uniform_(-stdv, stdv)
        self.pre_beta.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the forward transformation for the given input x.

        Args:
            x: input sample, shape [batch_size, dim]

        Returns:
            y: sample after forward tranformation, shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward tranformation, shape [batch_size]
        """
        B, D = x.shape

        ##########################################################
        softplus = torch.nn.Softplus()
        alpha = softplus(self.pre_alpha)
        beta = - alpha + softplus(self.pre_beta)

        def h(h_x):
            return 1 / (alpha + (torch.norm(h_x - self.x0)))

        def h_der(h_der_x):
            return h_der_x / torch.norm(h_der_x - self.x0)

        def radial(radial_x: Tensor) -> Tensor:
            h_result = h(radial_x)
            return radial_x +  beta *  h_result * (radial_x-self.x0)

        def determinant(det_x):
           e_1 = torch.pow(1 + beta * h(det_x), self.dim - 1)
           e_2 = 1 + beta * h(det_x) + beta * h_der(det_x) * torch.norm(x - self.x0)
           return torch.log(torch.abs(torch.prod(e_1 * e_2)))

        y = None
        log_det_jac = []
        for sample in range(B):
            current_sample = x[sample]
            radial_x = radial(current_sample)
            radial_x = radial_x[:, None]
            y = radial_x if y is None else torch.cat((y, radial_x), 1)
            det = determinant(current_sample)
            log_det_jac = [*log_det_jac, det.item()]
        log_det_jac = Tensor(log_det_jac)
        y = y.t()
        #########################################################

        assert y.shape == (B, D)
        assert log_det_jac.shape == (B,)

        return y, log_det_jac

    def inverse(self, y: Tensor) -> None:
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse tranformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse tranformation, shape [batch_size]
        """
        raise ValueError("The inverse tranformation is not known in closed form.")