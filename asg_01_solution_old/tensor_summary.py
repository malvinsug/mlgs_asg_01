import torch
x = torch.tensor(3., requires_grad=True)

a = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

y = a * x

z = y + b

#   x = torch.tensor(3., requires_grad=True)
print("Tensor x")
print(f'grad funtion = {x.grad_fn}')
print(f'is leaf = {x.is_leaf}')
print(x.requires_grad)

print("\nTensor y")
print(f'grad funtion = {y.grad_fn}')
print(f'is leaf = {y.is_leaf}')
print(y.requires_grad)

print("\nTensor z")
print(f'grad funtion = {z.grad_fn}')
print(f'is leaf = {z.is_leaf}')
print(z.requires_grad)