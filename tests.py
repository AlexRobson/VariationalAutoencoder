import unittest
import torch
from torch.autograd import Variable
from vae import *



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
		self.assertIs(X.shape[0],4)
#		self.assertAlmostEqual(X.tolist(), [-1.278, -0.4047, -0.4185, -1.8826])
		self.assertAlmostEqual(X[0], -1.278, places=2)

	def test_Encoder(self):
		encoder = Encoder()
		torch.manual_seed(0)
		params = {'batch_size': 10, 'test_batch_size': 10}
		train_loader, test_loader = vae_setup(params)
		X = next(iter(train_loader))
		mu, sigma = encoder.forward(Variable(X[0]))
		self.assertIs(mu.data.shape[0], 10)
		self.assertIs(mu.data.shape[1], 8)

if __name__ == '__main__':
	unittest.main()
