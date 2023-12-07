import os
import unittest
import torch

from PIL import Image
from matplotlib import pyplot as plt
from ray_nn.utils.ray_processing import HistToPointCloud
from sub_projects.ray_optimization.real_data import get_image, subtract_black, to_tensor


class TestRealData(unittest.TestCase):
    def setUp(self):
        self.path = 'testing/datasets/test_dataset'
        self.image_path = os.path.join(self.path, 'M03/10.bmp')
        self.image = get_image(self.image_path)
        self.black = get_image(os.path.join(self.path, 'black.bmp'))
        self.diff = subtract_black(self.image, self.black)
        self.tensor = to_tensor(self.diff)
        self.transform = HistToPointCloud()

        return
    
    def test_get_image(self):
        self.assertTrue(isinstance(self.image, Image.Image))
    
    def test_subtract_black(self):
        self.assertTrue(isinstance(self.diff, Image.Image))
        plt.imshow(self.diff)
        plt.savefig("diff.png")

    def test_to_tensor(self):
        plt.imshow(self.tensor.permute(1, 2, 0))
        plt.savefig("tens.png")
    
    def test_transform(self):
        self.transformed_tensor = self.transform(self.tensor, x_lims=torch.Tensor([[0,1]]), y_lims=torch.Tensor([[0,1]]))
        plt.scatter(self.transformed_tensor[0][0,:,0], y = self.transformed_tensor[0][0,:,1], c=self.transformed_tensor[1][0])
        plt.savefig("transf_tens.png")

if __name__ == "__main__":
    unittest.main()
