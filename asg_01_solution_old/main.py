from affine import *
from radial import *
from stack_flows import *

def train(model, dataset, batch_size=100, max_epochs=1000, frequency=250):
    """Train a normalizing flow model with maximum likelihood.

    Args:
        model: normalizing flow model. Flow or StackedFlows
        dataset: dataset containing data to fit. Dataset
        batch_size: number of samples per batch. int
        max_epochs: number of training epochs. int
        frequency: frequency for plotting density visualization. int
        
    Return:
        model: trained model. Flow or StackedFlows
        losses: loss evolution during training. list of floats
    """
    # Load dataset
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Train model
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    for epoch in range(max_epochs + 1):
        total_loss = 0
        for batch_index, (X_train) in enumerate(train_loader):
            ##########################################################
            # YOUR CODE HERE

            ##########################################################
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        losses.append(total_loss)
        
        if epoch % frequency == 0:
            print(f"Epoch {epoch} -> loss: {total_loss:.2f}")
            nf.plot_density(model, train_loader, device=device)
    
    return model, losses

#DATASET_1
dataset_1 = nf.CircleGaussiansDataset(n_gaussians=1, n_samples=500)
plt.figure(figsize=(4, 4))
plt.scatter(dataset_1.X[:,0], dataset_1.X[:,1], alpha=.05, marker='x', c='C1')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

##Affine
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

##Radial
transforms = [Radial().get_inverse().to(device) for _ in range(4)]
model = StackedFlows(transforms, base_dist='Normal').to(device)
model, losses = train(model, dataset_1, max_epochs=501)

# Plots
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
nf.plot_density(model, [], device=device)

#DATASET_2
dataset_2 = nf.CircleGaussiansDataset(n_gaussians=3, n_samples=400, variance=.4)
plt.figure(figsize=(4, 4))
plt.scatter(dataset_2.X[:,0], dataset_2.X[:,1], alpha=.05, marker='x', c='C1')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

transforms = [Affine().to(device)]
model = StackedFlows(transforms, base_dist='Normal').to(device)
model, losses = train(model, dataset_2, max_epochs=201)

# Plots
plt.plot(losses, marker='*')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
nf.plot_density(model, [], device=device)
nf.plot_samples(model)
transforms = [Radial().get_inverse() for _ in range(16)]
model = StackedFlows(transforms, base_dist='Normal').to(device)
model, losses = train(model, dataset_2, max_epochs=501, frequency=100)

# Plots
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
nf.plot_density(model, [], device=device)