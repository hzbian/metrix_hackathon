from ray_nn.data.lightning_data_module import DefaultDataModule
from ray_nn.nn.xy_hist_data_models import MetrixXYHistSurrogate
import glob
import torch
from ray_nn.nn.xy_hist_data_models import StandardizeXYHist
from ray_tools.simulation.torch_datasets import (
    BalancedMemoryDataset,
    HistDataset,
)
import glob

from scipy.stats import ttest_ind

@staticmethod
def significant_confidence_levels(group_A, group_B, confidence=0.95):
    ci = ttest_ind(group_A.flatten().cpu(), group_B.flatten().cpu(), equal_var=False).confidence_interval(confidence_level=confidence)
    confidence_interval = (ci.low.item(), ci.high.item())
    return not (confidence_interval[0] < 0. and confidence_interval[1] > 0.), confidence_interval

@staticmethod
def result_dict_to_latex(statistics_dict):
    if len(result_dict) < 4:
        alignment = "l" * len(statistics_dict)
        table_environment = "tabular"
    else:
        alignment = r"""*{"""+str(len(statistics_dict))+r"""}{>{\centering\arraybackslash}X}"""
        table_environment = "tabularx"
    
    if table_environment =="tabularx":
        text_width =  r"""{\textwidth}"""
    else:
        text_width = ""

    output_string = (
        r"""
    \begin{"""+table_environment+r"""}"""+text_width+r"""{p{2.5cm}|"""+
    alignment    
        + r"""}
    \hline"""
        + "\n"
    )
    scenarios = [k+r" $\pm\sigma$ (\acs{CI})" for k in statistics_dict.keys()]
    keys = ["Metric"] + scenarios
    output_string += " & ".join(keys) + r" \\" + "\n" 
    output_string += r"\hline" + "\n"
    
    model_keys = list(list(statistics_dict.values())[0].keys())

    for model_key in model_keys:
        model_row = [model_key]
        for (mean, std_dev, is_best, is_significant, p_value) in [v[model_key] for v in statistics_dict.values()]:
            model_row_element = f"{mean:.{4}f}"
            if is_best:
                model_row_element = r"\mathbf{" + model_row_element + r"}"
            model_row_element = r"$"+model_row_element+r" \pm "+f"{std_dev:.{4}f}$"
            if not is_best:
                model_row_element += f" ({p_value[0]:.1e}, {p_value[1]:.1e})"
                if is_significant:
                    model_row_element += r"$\dagger$"
            model_row += [model_row_element]
        output_string += " & ".join(model_row) + r" \\" + "\n"

    output_string += r"""\hline
    \end{"""+table_environment+r"""}"""
    return output_string
def evaluate_model_dict_to_result_dict(model_dict):
    result_dict = {}
    for scenario_name, scenario_subset in metrics_dict.items():
        result_dict[scenario_name] = {model_key: evaluate(model, scenario_subset) for model_key, model in model_dict.items()}
    return result_dict

@staticmethod
def significant_confidence_levels(group_A, group_B, confidence=0.99):
    ci = ttest_ind(group_A.flatten().cpu(), group_B.flatten().cpu(), equal_var=False).confidence_interval(confidence_level=confidence)
    confidence_interval = (ci.low.item(), ci.high.item())
    return not (confidence_interval[0] < 0. and confidence_interval[1] > 0.), confidence_interval


def statistics(result_dict):
    min_mean = float('inf')
    statistics_dict = {}
    for key, value in result_dict.items():
        mean = value.mean()
        statistics_dict[key] = (mean.item(), value.std().item())
        if mean < min_mean:
            min_mean_key = key
            min_mean = mean

    for key, value in result_dict.items():
         statistics_dict[key] =  statistics_dict[key] + (key==min_mean_key,) + significant_confidence_levels(value, result_dict[min_mean_key])
         diff = (result_dict[key] - result_dict[min_mean_key]).flatten().abs().cpu()
         mean = torch.mean(diff)
         std_dev = torch.std(diff)
    return statistics_dict

def model_paths_to_model_dict(model_paths):
    models_dict = {}
    for key, path in model_paths.items():
        models_dict[key] = MetrixXYHistSurrogate.load_from_checkpoint(
        checkpoint_path=path,
        #hparams_file="/path/to/experiment/version/hparams.yaml",
        map_location=None,
        )
    return models_dict
def evaluate(model, subset='good', load_len=5000):
    model.criterion = torch.nn.MSELoss(reduction='none')
    standardizer = model.standardizer
    output_list = []
    h5_files = list(glob.iglob('datasets/metrix_simulation/ray_emergency_surrogate_50+50+z+-30/histogram_*.h5'))
    sub_groups = ['parameters', 'histogram/ImagePlane', 'n_rays/ImagePlane']
    transforms=[lambda x: x[1:].float(), lambda x: standardizer(x.flatten().float()), lambda x: x.int()]
    dataset = HistDataset(h5_files, sub_groups, transforms, normalize_sub_groups=['parameters'], load_max=torch.ceil(torch.tensor(load_len) / len(h5_files)).int().item() )
    memory_dataset = BalancedMemoryDataset(dataset=dataset, load_len=load_len, min_n_rays=1, subset=subset)
    del dataset
    num_workers = 0
    datamodule = DefaultDataModule(dataset=memory_dataset, num_workers=num_workers, split=[0.,0., 1.])
    datamodule.prepare_data()
    
    datamodule.setup(stage="test")

    for x, y in datamodule.test_dataloader():
        with torch.no_grad():
            #y_hat = model(x.to(model.device))
            x = x.to(model.device)
            y = y.to(model.device)
            output_list.append(model.test_step((x,y)).mean(dim=-1))
    if len(output_list) > 0:
        output_tensor = torch.cat(output_list)
    return output_tensor

metrics_dict = {r"Nonempty \acs{MSE}":"good", r"Empty \acs{MSE}":"bad"}
model_paths = {"modelA": "outputs/xy_hist/i528cqk2/checkpoints/epoch=2-step=3.ckpt",
               "modelB": "outputs/xy_hist/i528cqk2/checkpoints/epoch=2-step=3.ckpt",
                "modelC": "outputs/xy_hist/i528cqk2/checkpoints/epoch=2-step=3.ckpt",
                "modelD": "outputs/xy_hist/i528cqk2/checkpoints/epoch=2-step=3.ckpt"
              }
model_dict = model_paths_to_model_dict(model_paths)

result_dict = evaluate_model_dict_to_result_dict(model_dict)
    
statistics_dict = {key: statistics(value) for key, value in result_dict.items()}
print(result_dict_to_latex(statistics_dict))