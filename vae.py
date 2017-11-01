# Load the data into memory
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets, transforms
from torchvision.utils import save_image

N_LATENT = 8


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


	# Taken from https://github.com/pytorch/examples/blob/master/mnist/main.py
class Encoder(nn.Module):
	"""
	This function is the neural networks encoding the (parameters of) the distribution that represents
	q(z | x). This is our 'oracle' distribution; a variational posterior. It is an approximate distribution to
	represent the real posterior p(z | x) the distribution about our latent variables of our data x.
	The optimal q is one that minimises the Kullback-Leibler distance:
		q*(z | x, \theta) = argmin_{\theta}{KL(q(z | x, \theta) || p(z | x)}

	It is a neural network that outputs the parameters \mu (x) and \sigma (x) that represent the distribution q(z | x)
	This is set-up as if a Normal distribution because it provides nice properties for when
	we arrive at the loss function (squared error), and thus the output of this network is a MU and SIGMA.

	These each have a size of BATCH_SIZE * N_LATENTS

	:return: \mu, \sigma
	"""

	def __init__(self):
		super(Encoder, self).__init__()
		self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
		self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
		self.fc1 = nn.Linear(10*20*20, 50)
		self.fc2 = nn.Linear(50, N_LATENT)


	def forward(self, x):
		#print(x.size())
		x = F.relu(self.conv1(x))
		#print(x.size())
		x = F.relu(self.conv2(x))
		#print(x.size())
		x = x.view(-1, 10*20*20)
		x = F.relu(self.fc1(x))
		mu = self.fc2(x)
		sigma = torch.exp(self.fc2(x))

		return mu, sigma


class Sampler(nn.Module):
	"""
	This function is defined explicstly to capture the sampling involved
	at the output of the encoder. The output of the encoder is Q(z | X). However, when
	following through the equations in variational autoencoders, the equation that represent
	log P(x) is built from two terms; a KullBack-Leibler distance of KL(Q(z | X) || P(z) )
	and a second term which is the built from an expectation over E{Q_z}[log P(X | z)].
	This stochastic layer captures this expectation, by sampling the output of Q(z | X).

	The secondi mportant aspect of this layer is the reparameterisation trick. In order to
	backprop through the output of the encoder, we need to take derivatives of N(MU, SIGMA),
	which is our estimate for Q(z | X). Therefore, we reparameterise. Instead of sampling
	from z ~ N(MU, SIGMA) we sample from \eps ~ N(0, 1) and then: z = MU + SIGMA * eps
	"""
	def __init__(self):
		super(Sampler, self).__init__()


	def forward(self, params):
		mu, sigma = params
		# Sample from N(0, 1)
		s = Variable(torch.normal(torch.zeros(mu.data.shape), torch.ones(mu.data.shape)))
		Z = mu + s * sigma
		return Z

class Decoder(nn.Module):
	"""
	This function is the network decoding that represents q(z | x) back into our observable space.
	Importantly, the input to this decoder step is a sample from ~N(MU, SIGMA),the output from the encoder.

	It provides a reconstruction of the observables

	:return:
	"""
	def __init__(self):
		super(Decoder, self).__init__()
		self.fc2 = nn.Linear(N_LATENT, 50)
		self.fc1 = nn.Linear(50, 10*20*20)
		self.conv2 = nn.ConvTranspose2d(10, 5, kernel_size=5)
		self.conv1 = nn.ConvTranspose2d(5, 1, kernel_size=5)

	def forward(self, x):
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc1(x))
		x = x.view(-1, 10, 20, 20)
		x = F.relu(self.conv2(x))
		x = F.sigmoid(self.conv1(x)) # Treat the output as a probability P(x | Z)

		return x


class VAEloss(nn.Module):
	"""
	The loss in the variational autoencoder is best expressed as the decomposition of two terms. The first can be
	described as a 'reconstruction error'. This is akin to the standard autoencoder where the recreation of the
	observables	X_dash, at the output of the decoder, relative to the input observables, X, is calculated.
	The second is the Kullback-Leibler regulariser KL(q(z | X) || p(z). This is the Kullback-leibler divergence
	between the encoder distribution q(z  | X), and the prior p(z). The first term is 'the expectation, over the
	encoders distributions, of the log likelihood of the i-th observation'.

	The KL term can be pulled straight from wikipedia:
	https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions

	The log-likliehood term is more complicated as it involves an expectation through a Monte Carlo estimate of the
	log-likelihood p(x | z)

	A likelihood p(x) can be modelled by considered the image data as a series of Bernoilli trials of success (white) = p
	and failure (black) = 1 - p. This specifies how to encoder log(p(X_i | z)). In this case, p(X_i | z) = p^y*(1-p)^(1-y).
	The log-term of this is then the cross-entropy: y * log(p) + (1-y)*log(1-p). Y here is the correct label, i.e. X[i,j]
	for the i, jth pixel, and p is the probability of the i,jth pixel.

	x: (N_BATCHSIZE, 28, 28)
	x_dash = (N_BATCHSIZE, 28, 28)

	The derivation for this can be seen in Appendix 2 of the Dietrich paper.

	:return: loss
	"""

	def __init__(self):
		super(VAEloss, self).__init__()


	def forward(self, X, X_dash, q_mu, q_sigma):
		Ndim = q_mu.data.shape[1]
		reconstruction = torch.log(nn.functional.binary_cross_entropy(X_dash, X))
		KL = 0.5 * torch.sum(1 + q_sigma.pow(2) + q_mu.pow(2) - torch.log(q_sigma.pow(2)), 1)
		loss = reconstruction - KL
		return loss


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.encoder = Encoder()
		self.sampler = Sampler()
		self.decoder = Decoder()

	def forward(self, X):
		return self.decoder(self.sampler((self.encoder(X))))


def update(X, model, loss, opt):

	# Perform a forward pass
	q_mu, q_sigma = model.encoder(X)
	X_dash = model(X)
	l = torch.sum(loss(X, X_dash, q_mu, q_sigma))
	print(l.data[0])
	opt.zero_grad()
	l.backward()
	opt.step()



# TESTING
if __name__=='__main__':
	params = {'batch_size': 100, 'test_batch_size': 100}
	train_loader, test_loader = vae_setup(params)
	model = Model()
	loss = VAEloss()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
	for i in range(1000):
		X = Variable(next(iter(train_loader))[0])
		X_dash = model(X)
		save_image(X.data, "./original/output_{0:0>3}.jpg".format(str(i)))
		save_image(X_dash.data, "./reconstructed/output_{0:0>3}.jpg".format(str(i)))
		update(X, model, loss, optimizer)

