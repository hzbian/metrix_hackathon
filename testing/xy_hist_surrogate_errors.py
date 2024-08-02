import glob
import tqdm
import matplotlib.pyplot as plt
import torch

from ray_nn.nn.xy_hist_data_models import MetrixXYHistSurrogate, StandardizeXYHist
from ray_tools.simulation.torch_datasets import MemoryDataset, RayDataset
from datasets.metrix_simulation.config_ray_emergency_surrogate import PARAM_CONTAINER_FUNC as params
from torch.utils.data import DataLoader
from ray_nn.data.transform import Select


model = MetrixXYHistSurrogate.load_from_checkpoint("outputs/xy_hist/i7sryekx/checkpoints/epoch=174-step=42782950.ckpt")
model.to(torch.device('cpu'))
model.compile()
model.eval()

load_len: int | None = None
h5_files = list(glob.iglob('datasets/metrix_simulation/ray_emergency_surrogate/selected/data_raw_*.h5'))
dataset = RayDataset(h5_files=h5_files,
                        sub_groups=['1e5/params',
                                    '1e5/ray_output/ImagePlane/histogram', '1e5/ray_output/ImagePlane/n_rays'], transform=Select(keys=['1e5/params', '1e5/ray_output/ImagePlane/histogram', '1e5/ray_output/ImagePlane/n_rays'], search_space=params(), non_dict_transform={'1e5/ray_output/ImagePlane/histogram': model.standardizer}))


memory_dataset = MemoryDataset(dataset=dataset, load_len=load_len)

train_dataloader = DataLoader(memory_dataset, batch_size=2048, shuffle=False, num_workers=0)

errors_list = []
with torch.no_grad():
    for par_input, label, _ in tqdm.tqdm(train_dataloader):
        out = model(par_input)
        label = label.flatten(start_dim=1)
        b = ((label - out)**2).mean(dim=1)
        errors_list.append(b)
errors_tensor = torch.cat(errors_list)

plt.hist(errors_tensor)
plt.savefig('outputs/dataset_errors_hist.png')
torch.save(errors_tensor, 'outputs/dataset_errors.pt')