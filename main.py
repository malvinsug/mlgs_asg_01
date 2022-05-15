import matplotlib.pyplot as plt

import nf_utils as nf
from affine import Affine
from device import device
from stacked_flows import StackedFlows
from train import train


dataset_1 = nf.CircleGaussiansDataset(n_gaussians=1, n_samples=500)

transforms = [Affine()]
model = StackedFlows(transforms, base_dist='Normal').to(device)
model, losses = train(model, dataset_1, max_epochs=201)

# Plots
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
nf.plot_density(model, [], device=device)
nf.plot_samples(model)

