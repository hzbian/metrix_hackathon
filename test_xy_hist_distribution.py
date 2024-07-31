import torch
import glob
import tqdm
import matplotlib.pyplot as plt
from ray_optim.plot import Plot
from ray_nn.data.transform import Select
from ray_nn.nn.xy_hist_data_models import MetrixXYHistSurrogate, StandardizeXYHist, HistSurrogateEngine
from ray_tools.base.backend import RayBackendDockerRAYUI
from ray_tools.base.transform import MultiLayer
from ray_tools.base.engine import RayEngine
from ray_nn.data.lightning_data_module import DefaultDataModule
from datasets.metrix_simulation.config_ray_emergency_surrogate import PARAM_CONTAINER_FUNC as params
from ray_tools.base.parameter import NumericalParameter, RayParameterContainer
from ray_tools.simulation.torch_datasets import BalancedMemoryDataset, MemoryDataset, RayDataset

def tensor_to_param_container(ten):
    param_dict = {}
    for i, (label, entry) in enumerate(params().items()):
        if label == 'U41_318eV.numberRays':
            param_dict[label] = entry
        else:
            value = ten[i-1]*(entry.value_lims[1]-entry.value_lims[0])+entry.value_lims[0]
            param_dict[label] = NumericalParameter(value.item())
            if value.item() < entry.value_lims[0] or value.item() > entry.value_lims[1]:
                if value.item() < entry.value_lims[0]:
                    value = torch.ones_like(value) * entry.value_lims[0]
                elif value.item() > entry.value_lims[1]:
                    value = torch.ones_like(value) * entry.value_lims[1]
                #raise Exception("Out of range. Minimum was {}, maximum {} but value {}. Tensor value was {}.".format(entry.value_lims[0], entry.value_lims[1], value.item(), ten[i-1].item()))
    return RayParameterContainer(param_dict)

engine = RayEngine(rml_basefile='rml_src/METRIX_U41_G1_H1_318eV_PS_MLearn_1.15.rml',
                                exported_planes=["ImagePlane"],
                                ray_backend=RayBackendDockerRAYUI(docker_image='ray-ui-service',
                                                                  docker_container_name='ray-ui-service-test',
                                                                  dockerfile_path='ray_docker/rayui',
                                                                  ray_workdir='/dev/shm/ray-workdir',
                                                                  verbose=False),
                                num_workers=-1,
                                as_generator=False)
surrogate_engine = HistSurrogateEngine(checkpoint_path="outputs/xy_hist/i7sryekx_copy/checkpoints/epoch=186-step=45716638.ckpt")

load_len: int | None = None
h5_files = list(glob.iglob('datasets/metrix_simulation/ray_emergency_surrogate/data_raw_*.h5'))
dataset = RayDataset(h5_files=h5_files,
                        sub_groups=['1e5/params',
                                    '1e5/ray_output/ImagePlane/histogram', '1e5/ray_output/ImagePlane/n_rays'], transform=Select(keys=['1e5/params', '1e5/ray_output/ImagePlane/histogram', '1e5/ray_output/ImagePlane/n_rays'], search_space=params(), non_dict_transform={'1e5/ray_output/ImagePlane/histogram': surrogate_engine.model.standardizer}))


memory_dataset = MemoryDataset(dataset=dataset, load_len=load_len)

unbal_datamodule = DefaultDataModule(dataset=memory_dataset, num_workers=4)
unbal_datamodule.prepare_data()
unbal_datamodule.setup(stage="test")
unbal_test_dl = unbal_datamodule.test_dataloader()

value_list = []
params_list = []
for i in tqdm.tqdm(unbal_test_dl):
    biggest = i[1].flatten(start_dim=1)
    biggest, _ = i[1].flatten(start_dim=1).max(dim=1)
    mask = biggest > 0.8
    value_list.append(biggest[mask])
    params_list.append(i[0][mask])
value_tensor = torch.cat(value_list)
params_tensor = torch.cat(params_list)
torch.save(value_tensor, 'values.pt')
torch.save(params_tensor, 'params.pt')
plt.hist(value_tensor)
plt.savefig('max_dist_hist.png')