# Load the data into memory
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Variable
import shutil
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image

N_LATENT = 8
debug = False


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
		self.fc1 = nn.Linear(28*28, 400)

		self.fc4 = nn.Linear(400, N_LATENT)
		self.fc5 = nn.Linear(400, N_LATENT)


	def forward(self, x):
		x = x.view(-1, 28*28)
		h = F.relu(self.fc1(x))
		mu = self.fc4(h)
		logsigma2 = self.fc5(h)
		mu = mu.view(-1, N_LATENT)
		logsigma2 = logsigma2.view(-1, N_LATENT)

		return mu, logsigma2


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
		mu, logsigma2 = params
		# Sample from N(0, 1)
		s = Variable(torch.normal(torch.zeros(mu.data.shape), torch.ones(mu.data.shape)), requires_grad=False)
		Z = mu + s * torch.exp(0.5 * logsigma2)
		return Z

class Decoder(nn.Module):
	"""
	This function is the network decoding that represents q(z | x) back into our observable space.
	Importantly, the input to this decoder step is a sample from ~N(MU, SIGMA),the output from the encoder.

	It provides a reconstruction of the observables. Model this as a Bernouilli distribution

	:return:
	"""
	def __init__(self):
		super(Decoder, self).__init__()
		self.fc1 = nn.Linear(N_LATENT, 400)
		self.fc2 = nn.Linear(400, 28*28)

	def forward(self, z):
		x = F.relu(self.fc1(z))
		x = F.sigmoid(self.fc2(x))
		x = x.view(-1, 1, 28, 28)

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
	The log-term of this is then the cross-entropy: y * log(p) + (1-y)*log(1-p)i.e. the binary-cross-entropy.
	Y here is the correct label, i.e. X[i,j] for the i, jth pixel, and p is the probability of the i,jth pixel.

	x: (N_BATCHSIZE, 28, 28)
	x_dash = (N_BATCHSIZE, 28, 28)

	The derivation for this can be seen in Appendix 2 of the Dietrich paper.

	:return: loss
	"""

	def __init__(self):
		super(VAEloss, self).__init__()


	def forward(self, X, X_dash, q_mu, q_logsigma2):
		reconstruction = nn.functional.binary_cross_entropy(X_dash, X, size_average=True)
		KL = - 0.5 * torch.sum(1 + q_logsigma2 - q_mu.pow(2) - q_logsigma2.exp(), 1) # Sum over Z dimension
		KL /= q_mu.data.shape[0] # Consistency with binary_cross_entropy, which averages minibatches (size_average=True)
		KL /= 784 # Normalise (28*28)

		loss = reconstruction + KL

		if debug:
			print("****")
			print("Reconstruction loss: {}".format(reconstruction.data[0]))
			print("Kullback-leibler loss: {}".format(KL.data[0]))

			for p, varname in zip((q_logsigma2, q_mu.pow(2), q_logsigma2.exp()), ('logsigma2', 'q_mu.pow(2)', 'sigma^2')):
				print("Term loss {}: {}".format(varname, torch.sum(p)))

			if torch.lt(KL,0).data.any():
				raise ValueError("The KL divergence is positive-definite. Calculated: {}".format(KL.data))

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
	q_mu, q_logsigma2 = model.encoder(X)
	X_dash = model(X)
	l = torch.sum(loss(X, X_dash, q_mu, q_logsigma2))
	if debug:
		if np.isnan(l.data[0]):
			raise("NaN detected")
	opt.zero_grad()
	l.backward()
	opt.step()
	return l.data[0]


def refresh():
	if os.path.isdir("./reconstructed"):
		shutil.rmtree("./reconstructed")
	if os.path.isdir("./original"):
		shutil.rmtree("./original")

	os.mkdir("./original")
	os.mkdir("./reconstructed")


# TESTING
if __name__=='__main__':
	params = {'batch_size': 100, 'test_batch_size': 100}
	train_loader, test_loader = vae_setup(params)
	model = Model()
	loss = VAEloss()
#	optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2, weight_decay=1)
#	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1)
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1)
	refresh()
	for epoch in range(int(5)):
		train_loss = 0
		for idx, (data, _) in enumerate(train_loader):
			X = Variable(data)
			X_dash = model(X)
			train_loss += update(X, model, loss, optimizer)

		if (epoch % 1) == 0:
			print("{}:{}".format(epoch, train_loss / len(train_loader.dataset)))
			save_image(X.data, "./original/output_{0:0>5}.jpg".format(str(epoch)))
			save_image(X_dash.data, "./reconstructed/output_{0:0>5}.jpg".format(str(epoch)))

