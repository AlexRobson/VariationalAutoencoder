import unittest
from vae import *


class TestLoaders(unittest.TestCase):

	def test_loaders(self):
		params = {'batch_size': 10, 'test_batch_size': 10}
		train_loader, test_loader = vae_setup(params)
		self.assertIs(type(train_loader),torch.utils.data.dataloader.DataLoader)



if __name__ == '__main__':
	unittest.main()
