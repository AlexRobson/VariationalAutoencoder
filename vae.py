# Load the data into memory
import torch
from torchvision import datasets, transforms




def vae_setup(params, datadir="../"):
	# Taken from https://github.com/pytorch/examples/blob/master/mnist/main.py
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(
			'../data', train=True, download=True,
		    transform=transforms.Compose([
		    transforms.ToTensor(),
		    transforms.Normalize((0.1307,), (0.3081,))
			])),
		batch_size=params["batch_size"], shuffle=True
		)
	test_loader = \
		torch.utils.data.DataLoader(
			datasets.MNIST('../data', train=False, transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
			])),
			batch_size=params["test_batch_size"], shuffle=True
		)

	return train_loader, test_loader


if __name__=='__main__':
	params = {'batch_size': 10, 'test_batch_size': 10}
	loaders = vae_setup(params)



