import matplotlib.pyplot as plt

import nf_utils as nf
from affine import Affine
from device import device
from radial import Radial
from stacked_flows import StackedFlows
from train import train


dataset_1 = nf.CircleGaussiansDataset(n_gaussians=1, n_samples=500)

def train_1():
    transforms = [Affine().get_inverse()]
    model = StackedFlows(transforms, base_dist='Normal').to(device)
    model, losses = train(model, dataset_1, max_epochs=501)

    k = model.transforms[0]
    # Plots
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    nf.plot_density(model, [], device=device)

def train__():
    transforms = [Radial().get_inverse()]
    model = StackedFlows(transforms, base_dist='Normal').to(device)
    model, losses = train(model, dataset_1, max_epochs=501)

    k = model.transforms[0]
    # Plots
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    nf.plot_density(model, [], device=device)
    nf.plot_samples(model)

def train_2():
    transforms = [Radial().get_inverse().to(device) for _ in range(4)]
    model = StackedFlows(transforms, base_dist='Normal').to(device)
    model, losses = train(model, dataset_1, max_epochs=501)

    # Plots
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    nf.plot_density(model, [], device=device)

train__()

