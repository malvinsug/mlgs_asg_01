#%load_ext autoreload
#%autoreload 2
import torch
from torch.optim import Adam
from matplotlib import pyplot as plt
from utils import get_mnist_data, get_device
from models import ConvNN, SmoothClassifier
from training_and_evaluation import train_model
from torch.nn.functional import cross_entropy

mnist_trainset = get_mnist_data(train=True)
mnist_testset = get_mnist_data(train=False)
device = get_device()

base_classifier = ConvNN().to(device)

sigma = 1
batch_size = 128
lr = 1e-3
epochs = 1

model = SmoothClassifier(base_classifier=base_classifier, num_classes=10, 
                         sigma=sigma)
opt = Adam(model.parameters(), lr=lr)

def loss_function(x, y, model):
    logits = model(x).cpu()
    loss = cross_entropy(logits, y)
    return loss, logits

losses, accuracies = train_model(model, mnist_trainset, batch_size, device,
                                 loss_function=loss_function, optimizer=opt)

torch.save(model.base_classifier.state_dict(), 
           "models/randomized_smoothing.checkpoint")

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