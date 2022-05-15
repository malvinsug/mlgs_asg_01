import nf_utils as nf
from affine import Affine
from stacked_flows import StackedFlows
from train import train

dataset_1 = nf.CircleGaussiansDataset(n_gaussians=1, n_samples=500)

transform = [Affine(), Affine()]
model = StackedFlows(transform)
train(model, dataset_1)

