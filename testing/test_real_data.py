import os
import unittest
import torch

from PIL import Image
from matplotlib import pyplot as plt
from ray_nn.utils.ray_processing import HistToPointCloud
from sub_projects.ray_optimization.real_data import SampleRandomWeightedHist, get_image, subtract_black, to_tensor, get_lims, clean_intensity


class TestRealData(unittest.TestCase):
    def setUp(self):
        self.path = 'testing/datasets/test_dataset'
        self.image_path = os.path.join(self.path, 'M03/10.bmp')
        self.image = get_image(self.image_path)
        self.black = get_image(os.path.join(self.path, 'black.bmp'))
        self.diff = subtract_black(self.image, self.black)
        self.tensor = to_tensor(self.diff).permute(0,2,1)
        self.transform = HistToPointCloud()
        self.transform_weight = SampleRandomWeightedHist()
        self.mm_per_pixel = 1.6 / 1000.
        self.x_dilation = 0.5
        self.y_dilation = 0.3
        self.x_lims = get_lims(self.x_dilation, 768, self.mm_per_pixel)
        self.y_lims = get_lims(self.y_dilation, 576, self.mm_per_pixel)
        self.grid, self.intensity = self.transform(self.tensor, x_lims=self.x_lims, y_lims=self.y_lims)
        self.cleaned_intensity, self.cleaned_indices = clean_intensity(self.intensity[0], threshold=0.02)
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
    
    def test_get_lims(self):
        self.assertAlmostEqual(self.x_lims[0][0].item(), self.x_dilation, places=4)
        self.assertAlmostEqual(self.x_lims[0][1].item(), self.x_dilation+1.2288, places=4)

    def test_transform(self):
        self.assertAlmostEqual(self.grid[0, :, 0].min().item(), self.x_lims[0,0].item(), places=2)
        self.assertAlmostEqual(self.grid[0, :, 0].max().item(), self.x_lims[0,1].item(), places=2)
        self.assertAlmostEqual(self.grid[0, :, 1].min().item(), self.y_lims[0,0].item(), places=2)
        self.assertAlmostEqual(self.grid[0, :, 1].max().item(), self.y_lims[0,1].item(), places=2)
        plt.scatter(self.grid[0,:,0], y = self.grid[0,:,1], c=self.intensity[0])
        plt.savefig("transf_tens.png")

    def test_clean_intensity(self):
        plt.scatter(self.grid[0,self.cleaned_indices,0], y = self.grid[0,self.cleaned_indices,1], c=self.cleaned_intensity)
        plt.xlim(self.x_lims[0])
        plt.ylim(self.y_lims[0])
        plt.savefig("clean.png")

    def test_transform_weight(self):
        scatter = self.transform_weight(hist=self.grid[:, self.cleaned_indices], pc_weights=self.cleaned_intensity.unsqueeze(0),
                                           num_rays=1000)
        plt.scatter(scatter[0, :, 0], scatter[0, :, 1], s=0.3, alpha=0.03)
        plt.xlim(self.x_lims[0])
        plt.ylim(self.y_lims[0])
        plt.savefig("transformed_weighted.png")
if __name__ == "__main__":
    unittest.main()
