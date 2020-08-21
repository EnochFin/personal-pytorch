import torch

# `autograd` is central to all NN in PyTorch
# provides auto differentiation for all operations on tensors
# backprop is defined by how the code is run

# tensor and function are interconnected and build an acyclic graph
# that encodes a complete history of computation

# setting `required_grad` to True will start tracking all operations
x = torch.ones(2, 2, requires_grad=True)
print(x)

# tensors that are tracking will have a .grad_fn that describes how the tensor was made
y = x + 2
print(y)
print(y.grad_fn)

# more operations to show the changes
z = y * y * 3
out = z.mean()
print(z, out)

# showing you can change if you're tracking the grad_fn
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)

a.requires_grad_(True)
print(a.requires_grad)

b = (a * a).sum()
print(b.grad_fn)

# print gradients
out.backward()
print(f'd(out)/dx: {x.grad}')


# jacobian matrix is apparently something like this:
# for a matrix that is m x n
#     (dy_1/dx_1 ... dy_1/dx_n)
# J = (...       ...       ...)
#     (dy_1/dx_n ... dy_m/dx_n)

# this is an example of a vector-Jacobian product
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
	y = y * 2
print(y)

# couldn't calculate using torch.autograd here so I have to pass it in through .backward()
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)


# can stop tracking on tensors that have .requires_grad=True by wrapping code, or using .detach()
# wrapping code
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)

# using .detach()
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
