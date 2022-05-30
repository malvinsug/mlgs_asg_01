#%load_ext autoreload
#%autoreload 2
import torch
from torch.optim import Adam
from matplotlib import pyplot as plt
from utils import get_mnist_data, get_device,save_variable,load_variable
from models import ConvNN
from training_and_evaluation import train_model, predict_model
from attacks import gradient_attack
from torch.nn.functional import cross_entropy
from typing import Tuple

mnist_trainset = get_mnist_data(train=True)
mnist_testset = get_mnist_data(train=False)
device = get_device()

model = ConvNN()
model.to(device)

epochs = 2
batch_size = 128
test_batch_size = 1000  # feel free to change this
lr = 1e-3

opt = Adam(model.parameters(), lr=lr)

attack_args = {'norm': "2", "epsilon": 5}

def loss_function(x: torch.Tensor, y: torch.Tensor, model: torch.nn.Module,  
                  **attack_args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loss function used for adversarial training. First computes adversarial 
    examples on the input batch via gradient_attack and then computes the 
    logits and the loss on the adversarial examples.
    Parameters
    ----------
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the 
       number of channels, and N is the image width/height.
        The input batch to certify.
    y: torch.Tensor of shape [B, 1].
        The labels of the input batch.
    model: torch.nn.Module
        The classifier to be evaluated.
    attack_args: 
        additional arguments passed to the adversarial attack function.
    
    Returns
    -------
    Tuple containing
        * loss_pert: torch.Tensor, scalar
            Mean loss obtained on the adversarial examples.
        * logits_pert: torch.Tensor, shape [B, K], K is the number of classes
            The logits obtained on the adversarial examples.
    """
    ##########################################################
    # YOUR CODE HERE
    B,C,N,_ = x.shape
    def get_x_grad(x,y,model):
        x.requires_grad = True
        logits_original = model.forward(x)
        loss_original = cross_entropy(logits_original,y)
        model.zero_grad()
        loss_original.backward()
        return x.grad.data

    # FGSM Attack
    x_grad = get_x_grad(x,y,model)
    epsilon = attack_args["epsilon"]
    norm = int(attack_args["norm"])
    normed_x_grad = torch.norm(epsilon*x_grad.sign(),p=norm,dim=1)
    perturbed_x = x + torch.reshape(normed_x_grad,(B,C,N,N))
    
    # Since we are using MNIST dataset,where range(0,1), we need to ensure that it stays inside the range.
    perturbed_x = torch.clamp(perturbed_x, 0, 1)

    # Forward pertubed image
    logits_pert = model.forward(perturbed_x)
    loss_pert = cross_entropy(logits_pert,y).mean()

    ##########################################################
    # Important: don't forget to call model.zero_grad() after creating the 
    #            adversarial examples.
    return loss_pert, logits_pert

losses, accuracies = train_model(model, mnist_trainset, batch_size, device,
                                 loss_function=loss_function, optimizer=opt, 
                                 loss_args=attack_args, epochs=epochs)
    
torch.save(model.state_dict(), "models/adversarial_training.checkpoint")

fig = plt.figure(figsize=(10,3))
plt.subplot(121)
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.subplot(122)
plt.plot(accuracies)
plt.xlabel("Iteration")
plt.ylabel("Training Accuracy")
plt.show()

clean_accuracy = predict_model(model, mnist_testset, batch_size, device,
                               attack_function=None)

perturbed_accuracy = predict_model(model, mnist_testset, test_batch_size, device, 
                                   attack_function=gradient_attack, 
                                   attack_args=attack_args)

print(clean_accuracy)
print(perturbed_accuracy)