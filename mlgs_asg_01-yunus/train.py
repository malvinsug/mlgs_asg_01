#%%

import torch

import nf_utils as nf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

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
            log_prob = torch.sum(model.log_prob(X_train))
            loss = - log_prob/batch_size
            ##########################################################

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        losses.append(total_loss)

        print(f"Epoch {epoch} -> loss: {total_loss:.2f}")
        if epoch % frequency == 0:
            #print(f"Epoch {epoch} -> loss: {total_loss:.2f}")
            nf.plot_density(model, train_loader, device=device)

    return model, losses