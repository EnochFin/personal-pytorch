# simple feed-forward network
# typical procedure
# - define NN that has some learnable params
# - iterate over dataset
# - process input through the network
# - compute the loss (how wrong we were)
# - propogate gradients back into the network's params
# - update the wights of the network
#   - usually done with a simple rule like weight = weight - learning_rate * gradient

import torch
import torch.nn as nn
import torch.nn.functional as F

# optimizers for weight updates
import torch.optim as optim

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		# 1 input image channel, 6 output channels, 3x3 square convolution
		# convolution: https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
		# kernel
		self.conv1 = nn.Conv2d(1, 6, 3)
		self.conv2 = nn.Conv2d(6, 16, 3)

		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16*6*6, 120) # 6x6 from image dimension
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# If the size is a quare, can only specify a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		# F.relu is an activation function
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:] # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

net = Net()
print(net)

# we only need to define forward, and autograd will generate the backward function
params = list(net.parameters())
print(len(params))
print(params[0].size())   # conv1's .weight

# putting in a random 32x32 input
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# zero the gradient buffers, and fill in with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

# Note: torch.nn will only take mini-batches of samples, nn.Conv2d will take a 4D Tensor of:
# - xSamples x nChannels x Height x Width
# use input.unsueeze(0) to add the extra dimensions when dealing with a single sample

# LOSS Function
output = net(input)
target = torch.randn(10)     # some target
target = target.view(1, -1)  # make it the same shape as the output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# make sure to zero out the gradients or else they will accumulate instead of replace when calling .backward
net.zero_grad()

print(f'conv1.bias.grad before: {net.conv1.bias.grad}')

loss.backward()

print(f'conv1.bias.grad after: {net.conv1.bias.grad}')

# now to update the weights
learning_rate = 0.01
for f in net.parameters():
	f.data.sub_(f.grad.data * learning_rate)

# setup optimizer
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# training (MAKE SURE TO ZERO GRAD)
optimizer.zero_grad()
output = net(input)
# calc loss
loss = criterion(output, target)
loss.backward()
# update
optimizer.setup()
