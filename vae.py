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

def encoder():
	"""
	This function is the neural networks encoding the (parameters of) the distribution that represents
	q(z | x). This is our 'oracle' distribution; a variational posterior. It is an approximate distribution to
	represent the real posterior p(z | x) the distribution about our latent variables of our data x.
	The optimal q is one that minimises the Kullback-Leibler distance:
	    q*(z | x, \theta) = argmin_{\theta}{KL(q(z | x, \theta) || p(z | x)}

	It is a neural network that outputs the parameters \mu (x) and \sigma (x) that represent the distribution q(z | x)
	This is set-up as if a Normal distribution because it provides nice properties for when
	we arrive at the loss function (squared error), and thus the output of this network is a MU and SIGMA.

	These have a size of BATCH_SIZE *



	:return: \mu, \sigma
	"""
	pass
	return MU, SIGMAs

def sample():
	"""
	This function is defined explicetly to capture the sampling involved when passing the distribution of the
	 variation posterior outputted from the encoder to the decoder.

	"""
	pass

def decoder():
	"""
	This function is the network decoding that represents q(z | x) back into our observable space.
	Importantly, the input to this decoder step is a sample from ~N(MU, SIGMA),the output from the encoder.

	It provides a reconstruction of the observables


	:return:
	"""
	pass

def loss():
	pass



if __name__=='__main__':
	params = {'batch_size': 10, 'test_batch_size': 10}
	loaders = vae_setup(params)






