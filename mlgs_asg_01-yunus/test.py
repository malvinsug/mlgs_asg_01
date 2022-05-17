import torch

import nf_utils as nf
from affine import Affine
from stacked_flows import StackedFlows
from train import train

dataset_1 = nf.CircleGaussiansDataset(n_gaussians=1, n_samples=500)


def test_affine_forwad():
    dataset_1 = nf.CircleGaussiansDataset(n_gaussians=1, n_samples=500)
    x = torch.from_numpy(dataset_1.X)

    affine_1 = Affine().forward(x)
    assert torch.equal(x, Affine().inverse(affine_1[0])[0])

def test_stack_log_prob():
    dataset_1 = nf.CircleGaussiansDataset(n_gaussians=1, n_samples=500)
    x = torch.from_numpy(dataset_1.X)

    transform = [Affine(), Affine()]
    stacked = StackedFlows(transform)
    result = stacked.log_prob(x)
    assert result is not None

def test_stack_rsample():
    transform = [Affine(), Affine()]
    stacked = StackedFlows(transform)
    stacked.rsample(10)

def test_train():
    transform = [Affine(), Affine()]
    model = StackedFlows(transform)
    train(model, dataset_1)
    assert 1 is not None


