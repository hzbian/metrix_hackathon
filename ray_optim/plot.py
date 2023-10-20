from typing import List, Optional, Tuple
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
import torch

from ray_tools.base.parameter import MutableParameter, NumericalParameter, RayParameterContainer

mpl.rcParams['text.color'] = 'dimgray'
mpl.rcParams['axes.labelcolor'] = 'dimgray'
mpl.rcParams['xtick.color'] = 'dimgray'
mpl.rcParams['ytick.color'] = 'dimgray'
mpl.rcParams['axes.edgecolor'] = 'dimgray'
plt.switch_backend("Agg")


class Plot:
    @staticmethod
    def plot_data(
        pc_supp: List[torch.Tensor],
        pc_weights: Optional[List[torch.Tensor]] = None,
        epoch: Optional[int] = None,
    ) -> Figure:
        pc_supp = [v.detach().cpu() for v in pc_supp]
        pc_weights = (
            None if pc_weights is None else [v.detach().cpu() for v in pc_weights]
        )
        fig, axs = plt.subplots(len(pc_supp), pc_supp[0].shape[0], squeeze=False)
        for i, column in enumerate(pc_supp):
            for j, line in enumerate(column):
                axs[i, j].scatter(line[:, 0], line[:, 1], s=0.5, alpha=0.5)
        if epoch is not None:
            fig.suptitle("Epoch " + str(epoch))
        if len(column) > 1:
            fig.supxlabel('Shifting in Ray Direction')
        if len(pc_supp) > 1:
            fig.supylabel('Varying Parameters')
        fig.tight_layout()
        return fig

    @staticmethod
    def compensation_plot(
        compensated: List[torch.Tensor],
        target: List[torch.Tensor],
        without_compensation: List[torch.Tensor],
        epoch: Optional[int] = None,
    ) -> Figure:
       xlim_min = min([entry[0, :, 0].min().item() for entry in target])
       xlim_max = max([entry[0, :, 0].max().item() for entry in target])
       ylim_min = min([entry[0, :, 1].min().item() for entry in target])
       ylim_max = max([entry[0, :, 1].max().item() for entry in target])
       xlim = (xlim_min, xlim_max)
       ylim = (ylim_min, ylim_max)
       fig = Plot.fixed_position_plot(compensated=compensated, target=target, without_compensation=without_compensation, xlim=xlim, ylim=ylim, epoch=epoch)
       return fig

    @staticmethod
    def fixed_position_plot(
        compensated: list[torch.Tensor],
        target: list[torch.Tensor],
        without_compensation: list[torch.Tensor],
        xlim,
        ylim,
        epoch: Optional[int] = None,
    ) -> Figure:
        y_label = ["Uncompensated", "Observed", "Compensated"]
        suptitle = "Epoch " + str(epoch) if epoch is not None else None
        return Plot.fixed_position_plot_base(
            [without_compensation, target, compensated], xlim, ylim, y_label, suptitle
        )
    
    @staticmethod
    def normalize_parameters(
        parameters: RayParameterContainer, search_space: RayParameterContainer
    ) -> RayParameterContainer:
        normalized_parameters = RayParameterContainer()
        for key, value in search_space.items():
            if isinstance(value, MutableParameter):
                normalized_parameters[key] = NumericalParameter(
                    (parameters[key].get_value() - value.value_lims[0])
                    / (value.value_lims[1] - value.value_lims[0])
                )
        return normalized_parameters
    
    @staticmethod
    def fixed_position_plot_base(
        tensor_list_list: List[List[torch.Tensor]],
        xlim: Tuple[float],
        ylim: Tuple[float],
        ylabel,
        suptitle: Optional[str] = None,
    ) -> Figure:
        
        fig, axs = plt.subplots(
            len(tensor_list_list),
            len(tensor_list_list[0]),
            squeeze=False,
            gridspec_kw={"wspace": 0, "hspace": 0},
            figsize=(len(tensor_list_list[0])*2, len(tensor_list_list)*2),
            sharex=True, sharey=True,
        )
        for idx_list_list in range(len(tensor_list_list)):
            for beamline_idx in range(len(tensor_list_list[0])):
                element = tensor_list_list[idx_list_list][beamline_idx]
                xlim_half = (xlim[1]-xlim[0]) / 2
                xlim_middle = xlim[0]+xlim_half
                axs[idx_list_list, beamline_idx].set_xlim(xlim_middle-xlim_half*1.2, xlim_middle+xlim_half*1.2)
                ylim_half = (ylim[1]-ylim[0]) / 2
                ylim_middle = ylim[0]+ylim_half
                axs[idx_list_list, beamline_idx].set_ylim(ylim_middle-ylim_half*1.2, ylim_middle+ylim_half*1.2)
                axs[idx_list_list, beamline_idx].xaxis.set_major_locator(
                        plt.NullLocator()
                )
                axs[idx_list_list, beamline_idx].yaxis.set_major_locator(
                        plt.NullLocator()
                )
                axs[idx_list_list, beamline_idx].set_xticks(xlim)
                axs[idx_list_list, beamline_idx].set_yticks(ylim)
                axs[idx_list_list, beamline_idx].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                axs[idx_list_list, beamline_idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                axs[idx_list_list, beamline_idx].set_aspect("equal")
                axs[idx_list_list, beamline_idx].tick_params(axis='both', length=0.)
                axs[idx_list_list, beamline_idx].grid(linestyle = "dashed", alpha = 0.5)
                axs[idx_list_list, beamline_idx].scatter(
                    element[0, :, 0], element[0, :, 1], s=1.0, alpha=0.5, linewidths=0.4
                )

            axs[idx_list_list, 0].set_ylabel(ylabel[idx_list_list])
        if len(tensor_list_list[0]) > 1:
            fig.supxlabel('Varying Parameters')
        if suptitle is not None:
            fig.suptitle(suptitle) 
        fig.set_dpi(200)
        return fig

    @staticmethod
    def plot_param_comparison(
        predicted_params: RayParameterContainer,
        search_space: RayParameterContainer,
        epoch: int,
        real_params: Optional[RayParameterContainer] = None,
        omit_labels: Optional[List[str]] = None,
    ) -> Figure:
        if omit_labels is None:
            omit_labels = []
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.set_ylim([-0.5, 0.5])
        normalized_predicted_params = [
            param.get_value() - 0.5
            for param in Plot.normalize_parameters(
                predicted_params, search_space
            ).values()
        ]
        if real_params is not None:
            normalized_real_params = [
                param.get_value() - 0.5
                for param in Plot.normalize_parameters(
                    real_params, search_space
                ).values()
            ]
            len_params = len(normalized_real_params)
            ax2 = ax.twinx()
            ax2.set_ylim([0, 1])
            ax2.bar(
                [i for i in range(len_params)],
                [
                    abs(normalized_real_params[i] - normalized_predicted_params[i])
                    for i in range(len_params)
                ],
                color="tab:red",
                alpha=0.2,
                label="Difference"
            )
            ax2.set_ylabel('Difference', color='#d6272880')
            ax2.tick_params(axis='y', labelcolor='#d6272880', color='#d6272880')
            ax.stem(
                normalized_real_params,
                label="Real Parameters",
            )
            ax.stem(
            normalized_predicted_params,
            linefmt="orange",
            markerfmt="o",
            label="Predicted Parameters",
            )
        param_labels = [
            param_key
            for param_key, _ in predicted_params.items()
            if param_key not in omit_labels
        ]
        ax.set_xticks(range(len(param_labels)))
        ax.set_xticklabels(param_labels, rotation=90)
        fig.legend()
        plt.subplots_adjust(bottom=0.3)
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Normalized Compensation")
        fig.suptitle("Epoch " + str(epoch))
        return fig
