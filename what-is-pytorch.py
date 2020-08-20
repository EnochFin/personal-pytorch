# This follows the tutorial found on
# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-ensor-tutorial-py

from __future__ import print_function
import torch
import numpy as np

# uninitialized matricies don't contain definite known values, until it's used

# when an uninit matrix is created, whatever values were in the allocated memory will
# appear as the initial values
x = torch.empty(5, 3)
print(f'empty: {x}')
# this outputs data that could be anything

x = torch.rand(5, 3)
print(f'random: {x}')

x = torch.zeros(5, 3, dtype=torch.long)
print(f'zeroes: {x}')

# create a tensor from hard-coded data
x = torch.tensor([5.5, 3])
print(f'hard-coded: {x}')

# new_* methods take in sizes
x = x.new_ones(5, 3, dtype=torch.double)
print(f'new_*: {x}')

# use randn_like to create a random tensor with the same size as the tensor passed in
x = torch.randn_like(x, dtype=torch.float)
print(f'randn_like: {x}')

# size() on a tensor returns a python Tuple for the size of the tensor
print(f'.size(): {x.size()}')

# you can add in pyTorch the same way you can in python
y = torch.ones(5, 3)
print(f'x + y: {x + y}')

# or using a function
print(f'torch.add(): {torch.add(x, y)}')

# .add() supports out variables
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(f'out variable: {result}')

# you can use inplace addition if you like mutability
y.add_(x)
print(f'in-place addition: {y}')
# in-place functions are post-fixed with an `_`

# can use numpy indexing :D
print(f'list interpolation: {x[:, 1]}')

# resize tensors using torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) # -1 in a view will be calculated based on other dimensions
print(f'view(): {x.size()} {y.size()} {z.size()}')

# tensors that have one element can be retrieved as a python number with .item()
x = torch.randn(1)
print(f'single-element tensor: {x}')
print(f'.item(): {x.item()}')
# read https://pytorch.org/docs/stable/torch.html for more functions

print(f'NumPy <-> Tensor')
# NumPy <-> Tensor
a = torch.ones(5)
b = a.numpy()
print(f'numpy copy of tensor: {b}')

# modifying the tensor will change the numpy array too
a.add_(1)
print(f'added to tensor: {a}')
print(f'numpy arr: {b}')

# now the other way
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(f'added to numpy: {a}')
print(f'tensor: {b}')

# tensors can be moved onto any device using .to()
if torch.cuda.is_available():
	device = torch.device("cuda") # CUDA device object
	y = torch.ones_like(x, device=device)
	x = x.to(device)
	z = x + y
	print(f'computed on the cuda device: {z}')
	print(z.to("cpu", torch.double))
