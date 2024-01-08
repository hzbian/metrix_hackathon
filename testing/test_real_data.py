import os
from pathlib import Path
import unittest
import torch
from PIL import Image
from matplotlib import pyplot as plt
from ray_nn.utils.ray_processing import HistToPointCloud
from ray_tools.base.transform import Histogram
from sub_projects.ray_optimization.real_data import SampleRandomWeightedHist, get_image, subtract_black, to_tensor, get_lims, clean_intensity, tensor_to_ray_output


class TestRealData(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
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
        self.threshold = 0.02
        self.cleaned_intensity, self.cleaned_indices = clean_intensity(self.intensity[0], threshold=self.threshold)
        self.num_scatter_rays = 1000000
        self.scatter = self.transform_weight(hist=self.grid[:, self.cleaned_indices], pc_weights=self.cleaned_intensity.unsqueeze(0),
                                           num_rays=self.num_scatter_rays)
        self.ray_output = tensor_to_ray_output(self.scatter)
        self.hist = Histogram(n_bins=768, x_lims=(self.x_lims[0][0].item(), self.x_lims[0][1].item()), y_lims=(self.y_lims[0][0].item(), self.y_lims[0][1].item()), n_bins_y=576)
        return
    
    def test_get_image(self):
        self.assertTrue(isinstance(self.image, Image.Image))
    
    def test_subtract_black(self):
        self.assertTrue(isinstance(self.diff, Image.Image))
        self.assertGreater(to_tensor(self.image).sum().item(), self.tensor.sum().item())

    def test_to_tensor(self):
        self.assertTrue(isinstance(self.tensor, torch.Tensor))
    
    def test_get_lims(self):
        self.assertAlmostEqual(self.x_lims[0][0].item(), self.x_dilation, places=4)
        self.assertAlmostEqual(self.x_lims[0][1].item(), self.x_dilation+self.mm_per_pixel*768, places=4)
        self.assertAlmostEqual(self.y_lims[0][0].item(), self.y_dilation, places=4)
        self.assertAlmostEqual(self.y_lims[0][1].item(), self.y_dilation+self.mm_per_pixel*576, places=4)

    def test_transform(self):
        self.assertAlmostEqual(self.grid[0, :, 0].min().item(), self.x_lims[0,0].item(), places=2)
        self.assertAlmostEqual(self.grid[0, :, 0].max().item(), self.x_lims[0,1].item(), places=2)
        self.assertAlmostEqual(self.grid[0, :, 1].min().item(), self.y_lims[0,0].item(), places=2)
        self.assertAlmostEqual(self.grid[0, :, 1].max().item(), self.y_lims[0,1].item(), places=2)
        center_of_mass = TestRealData.first_center_of_mass(self.grid, self.intensity)
        self.assertLess(((center_of_mass - torch.Tensor([0.0014, 0.0010]))).mean().item(), 1.49e-5)

    def test_clean_intensity(self):
        self.assertGreater(self.cleaned_intensity.min().item(), self.threshold)
        center_of_mass = TestRealData.first_center_of_mass(self.grid, self.intensity)
        center_of_mass_cleaned = TestRealData.first_center_of_mass(self.grid[:,self.cleaned_indices,:], self.cleaned_intensity)
        self.assertLess(((center_of_mass - center_of_mass_cleaned)**2).mean().item(), 0.00063)

    def test_transform_weight(self):
        self.assertEqual(self.scatter.shape[1], self.num_scatter_rays)

    def test_closed_loop(self):
        hist_out: torch.Tensor = self.hist(self.ray_output)['histogram']
        mse: float = TestRealData.normalized_mse(hist_out, self.tensor)
        self.assertLess(mse, 1.473e-5)
    
    def test_plot(self):
        hist_out: torch.Tensor = self.hist(self.ray_output)['histogram']
        plt.clf()
        plt.imshow(hist_out.T)
        out_dir = "outputs/test_real_data/"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "test_plot.png"))
    @staticmethod
    def first_center_of_mass(t: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (t[0] * weights[0].unsqueeze(-1)).mean(dim=(0))
    
    @staticmethod
    def normalized_mse(a: torch.Tensor, b: torch.Tensor) -> float:
        a = (a - a.min()) / (a.max() - a.min())
        b = (b - b.min()) / (b.max() - b.min())
        return ((a-b)**2).mean().item()
    
if __name__ == "__main__":
    unittest.main()
