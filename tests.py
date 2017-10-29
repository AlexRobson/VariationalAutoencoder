import unittest
import torch
from torch.autograd import Variable
from vae import *
import numpy as np


class TestVAE(unittest.TestCase):


	def test_loaders(self):
		params = {'batch_size': 10, 'test_batch_size': 10}
		train_loader, test_loader = vae_setup(params)
		self.assertIs(type(train_loader),torch.utils.data.dataloader.DataLoader)

	def test_Sampler(self):
		S = Sampler()
		self.assertIsInstance(S, Sampler)

		torch.manual_seed(0)
		X = S.forward(torch.Tensor([0,0,0,0]), torch.Tensor([1,1,1,1]))
		self.assertIs(X.data.shape[0],4)
#		self.assertAlmostEqual(X.tolist(), [-1.278, -0.4047, -0.4185, -1.8826])
		self.assertAlmostEqual(X[0], -1.278, places=2)

	def test_Encoder(self):
		encoder = Encoder()
		torch.manual_seed(0)
		params = {'batch_size': 23, 'test_batch_size': 23}
		train_loader, test_loader = vae_setup(params)
		X = next(iter(train_loader))
		mu, sigma = encoder.forward(Variable(X[0]))
		self.assertIs(mu.data.shape[0], 23)
		self.assertIs(mu.data.shape[1], 8)

	def test_Decoder(self):
		decoder = Decoder()
		torch.manual_seed(0)
		params = {'batch_size': 23, 'test_batch_size': 23}
		Z_shape = (23, 8)
		z_sample = torch.randn(Z_shape)
		X_dash = decoder.forward(Variable(z_sample))
		self.assertIs(X_dash.data.shape[0], 23)
		self.assertIs(X_dash.data.shape[2], 28)
		self.assertIs(X_dash.data.shape[3], 28)

	def test_Sampler(self):
		S = Sampler()
		Z_shape = (23, 8)
		mu = Variable(torch.randn(Z_shape))
		sigma = Variable(torch.randn(Z_shape))
		z_sample = S.forward((mu, sigma))
		self.assertSequenceEqual(tuple(z_sample.data.shape), Z_shape)

	def test_Loss(self):
		dims = (27, 1, 28, 28)
		nlatent = 8
		params = {'batch_size': dims[0], 'test_batch_size': dims[0]}
		train_loader, test_loader = vae_setup(params)
		X = Variable(next(iter(train_loader))[0])
		X_dash = Variable(torch.rand(dims))
		q_mu = Variable(torch.rand((dims[0], nlatent)))
		q_sigma = Variable(torch.rand(dims[0], nlatent))
		minibatch_loss = loss(X, X_dash, q_mu, q_sigma)

		outshape = minibatch_loss.data.shape
		self.assertIs(outshape[0], 27)
		self.assertIs(len(outshape), 1)

	def test_Model(self):

		encoder = Encoder()
		params = {'batch_size': 23, 'test_batch_size': 23}
		train_loader, test_loader = vae_setup(params)
		X = Variable(next(iter(train_loader))[0])
		X_dash = model(X)
		q_mu, q_sigma = encoder.forward(X)
		l = loss(X, X_dash, q_mu, q_sigma)



if __name__ == '__main__':
	unittest.main()
