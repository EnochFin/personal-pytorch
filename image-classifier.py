# Steps to training an image classifier

# 1. Load and normalize the CIFAR10 set
# 2. Define a convolutional Neural Network
# 3. Define loss func
# 4. Train the netowkr
# 5. Test the network

# 1. Loading and normalizing the CIFAR10 dataset

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
	img = img / 2 + 0.5 # unnormalize
	torch.Tensor.cpu() # figure out why this doesn't work
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

# get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

# show images
#imgshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 2. Define the Neural Network
class Net(nn.Module):
	def __init__(self):
		# call base ctor
		super(Net, self).__init__()
		# first convolution
		self.conv1 = nn.Conv2d(3, 6, 5)
		# max pool
		self.pool = nn.MaxPool2d(2, 2)
		# second convolution
		self.conv2 = nn.Conv2d(6, 16, 5)
		# linear activation reduction
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net = Net()
net.to(device)

# 3. Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. train the network
should_train = True
PATH = './cifar_net.pth'

if should_train:
	for epoch in range(2):

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get inputs from the training set
			inputs, labels = data[0].to(device), data[1].to(device)
			print(inputs)
			print(labels)
			print(data)
			# Zero the parameter gradients
			optimizer.zero_grad()

			# run forward, backward, optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print stats
			running_loss += loss.item()
			if i % 2000 == 1999:
				print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
				running_loss = 0.0
				print('inputs')
				print(inputs)
				print('labels')
				print(labels)
				print('data')
				print(data)
	torch.save(net.state_dict(), PATH)
	print('Done training!')
else:
	net = Net()
	net.load_state_dict(torch.load(PATH))
	net.to(device)


# 5. test the network
dataiter = iter(testloader)
val = dataiter.next()
images, labels = val[0].to(device), val[1].to(device)

# print images
#imshow(torchvision.utils.make_grid(images.cpu()))
print(f'GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# make predictions
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print(f'Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# test against whole test set
correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		images, labels = data[0].to(device), data[1].to(device)
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print(f'accuracy against whole test set: {100 * correct / total}')

# see performance per class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
	for data in testloader:
		images, labels = data[0].to(device), data[1].to(device)
		outputs = net(images)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()
		for i in range(4):
			label = labels[i]
			class_correct[label] += c[i].item()
			class_total[label] += 1
for i in range(10):
	print('accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))











