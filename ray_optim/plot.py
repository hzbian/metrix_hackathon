from typing import List, Optional, Tuple
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import torch

from ray_tools.base.parameter import (
    MutableParameter,
    NumericalParameter,
    RayParameterContainer,
)

mpl.rcParams["text.color"] = "dimgray"
mpl.rcParams["axes.labelcolor"] = "dimgray"
mpl.rcParams["xtick.color"] = "dimgray"
mpl.rcParams["ytick.color"] = "dimgray"
mpl.rcParams["axes.edgecolor"] = "dimgray"
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
        fig, axs = plt.subplots(
            len(pc_supp), pc_supp[0].shape[0], squeeze=False, layout="constrained"
        )
        for i, column in enumerate(pc_supp):
            for j, line in enumerate(column):
                axs[i, j].scatter(line[:, 0], line[:, 1], s=0.5, alpha=0.5)
        if epoch is not None:
            fig.suptitle("Epoch " + str(epoch))
        if len(column) > 1:
            fig.supxlabel("Shifting in Ray Direction")
        if len(pc_supp) > 1:
            fig.supylabel("Varying Parameters")
        return fig
    
    @staticmethod
    def fancy_ray(data: List[torch.Tensor]):
        fig, ax = plt.subplots(len(data), len(data[0]), squeeze=False, figsize=(8,8), subplot_kw=dict(projection='3d'), layout="constrained")
        for i, row in enumerate(data):
            for j, column in enumerate(row):
                zdata = column.flatten(0, 1)[:, 0]
                ydata = column.flatten(0, 1)[:, 1]
                xdata = torch.cat([torch.ones_like(column[0, :, 0]) * i for i in range(column.shape[0])])

                ax[0][j].scatter(xdata, ydata, zdata, alpha=0.5, linewidths=0., s=8.0)
                #ax[i][j].view_init(9, -60) 
        return fig
    
    @staticmethod
    def compensation_plot(
        compensated: List[torch.Tensor],
        target: List[torch.Tensor],
        without_compensation: List[torch.Tensor],
        epoch: Optional[int] = None,
        covariance_ellipse: bool = True,
    ) -> Figure:
        xlim_min = [entry[0, :, 0].min().item() for entry in target]
        xlim_max = [entry[0, :, 0].max().item() for entry in target]
        ylim_min = [entry[0, :, 1].min().item() for entry in target]
        ylim_max = [entry[0, :, 1].max().item() for entry in target]
        xlim = (xlim_min, xlim_max)
        ylim = (ylim_min, ylim_max)
        fig = Plot.fixed_position_plot(
            compensated=compensated,
            target=target,
            without_compensation=without_compensation,
            xlim=xlim,
            ylim=ylim,
            epoch=epoch,
            covariance_ellipse=covariance_ellipse,
        )
        return fig

    @staticmethod
    def fixed_position_plot(
        compensated: list[torch.Tensor],
        target: list[torch.Tensor],
        without_compensation: list[torch.Tensor],
        xlim,
        ylim,
        epoch: Optional[int] = None,
        covariance_ellipse: bool = True,
    ) -> Figure:
        y_label = ["Uncompensated", "Observed", "Compensated"]
        suptitle = "Epoch " + str(epoch) if epoch is not None else None
        return Plot.fixed_position_plot_base(
            [without_compensation, target, compensated],
            xlim,
            ylim,
            y_label,
            suptitle,
            covariance_ellipse,
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
    def get_lims_per_row(
        xlim: Tuple[float] | Tuple[List[float]],
        ylim: Tuple[float] | Tuple[List[float]],
        row_length: int,
    ):
        if isinstance(xlim[0], float):
            xlim_min = [xlim[0] for _ in range(row_length)]
            xlim_max = [xlim[1] for _ in range(row_length)]
        else:
            xlim_min = xlim[0]
            xlim_max = xlim[1]
        if isinstance(ylim[0], float):
            ylim_min = [ylim[0] for _ in range(row_length)]
            ylim_max = [ylim[1] for _ in range(row_length)]
        else:
            ylim_min = ylim[0]
            ylim_max = ylim[1]
        return xlim_min, xlim_max, ylim_min, ylim_max

    @staticmethod
    def scale_interval(minimum: float, maximum: float, scale: float):
        interval_half = (maximum - minimum) / 2
        interval_middle = minimum + interval_half
        return (
            interval_middle - interval_half * scale,
            interval_middle + interval_half * scale,
        )

    @staticmethod
    def fixed_position_plot_base(
        tensor_list_list: List[List[torch.Tensor]],
        xlim: Tuple[float] | Tuple[List[float]],
        ylim: Tuple[float] | Tuple[List[float]],
        ylabel,
        suptitle: Optional[str] = None,
        covariance_ellipse: bool = True,
        draw_all_ellipses_rows: Tuple[int] = (1,),
    ) -> Figure:
        fig, axs = plt.subplots(
            len(tensor_list_list),
            len(tensor_list_list[0]),
            squeeze=False,
            gridspec_kw={"wspace": 0, "hspace": 0},
            figsize=(len(tensor_list_list[0]) * 2, len(tensor_list_list) * 2),
            sharex=False,
            sharey=False,
            layout="compressed",
        )
        xlim_min, xlim_max, ylim_min, ylim_max = Plot.get_lims_per_row(xlim, ylim, len(tensor_list_list[0]))
        fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)
        engine = fig.get_layout_engine()
        engine.set(rect=(0.1, 0.0, 0.8, 1.0))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx_list_list in range(len(tensor_list_list)):
            color = colors[idx_list_list % len(colors)]
            for beamline_idx in range(len(tensor_list_list[0])):
                ax = axs[idx_list_list, beamline_idx]
                element = tensor_list_list[idx_list_list][beamline_idx]
                ax.set_xlim(
                    Plot.scale_interval(
                        xlim_min[beamline_idx], xlim_max[beamline_idx], 1.2
                    )
                )
                ax.set_ylim(
                    Plot.scale_interval(
                        ylim_min[beamline_idx], ylim_max[beamline_idx], 1.2
                    )
                )
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())
                if isinstance(xlim[0], float):
                    ax.set_xticks(xlim)
                    ax.set_yticks(ylim)
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.set_aspect("equal")
                ax.tick_params(axis="both", length=0.0)
                ax.grid(linestyle="dashed", alpha=0.5)
                alpha = 0.5 if not covariance_ellipse else 0.25
                scatter_color = color if covariance_ellipse else None
                ax.scatter(
                    element[0, :, 0],
                    element[0, :, 1],
                    s=1.0,
                    alpha=alpha,
                    linewidths=0.4,
                    color=scatter_color,
                )
                if covariance_ellipse & (element.shape[1] > 0):
                    if idx_list_list in draw_all_ellipses_rows:
                        for e in range(len(tensor_list_list)):
                            if e != idx_list_list:
                                other_ellipse_color = colors[e % len(colors)]
                                Plot.confidence_ellipse(
                                    tensor_list_list[e][beamline_idx][0, :, 0],
                                    tensor_list_list[e][beamline_idx][0, :, 1],
                                    ax,
                                    color=other_ellipse_color,
                                    alpha=0.5,
                                )
                    Plot.confidence_ellipse(
                        element[0, :, 0], element[0, :, 1], ax, color=color
                    )

            axs[idx_list_list, 0].set_ylabel(ylabel[idx_list_list], fontsize=12)
            if covariance_ellipse:
                axs[idx_list_list, 0].yaxis.label.set_color(color)
        if len(tensor_list_list[0]) > 1:
            fig.supxlabel("Varying Parameters")
        if suptitle is not None:
            fig.suptitle(suptitle)
        fig.set_dpi(200)
        return fig

    @staticmethod
    def confidence_ellipse(
        x, y, ax, color, n_std=3.0, facecolor="none", alpha=1.0, **kwargs
    ):
        """
        Create a plot of the covariance confidence ellipse of `x` and `y`

        See how and why this works: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

        This function has made it into the matplotlib examples collection:
        https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

        Or, once matplotlib 3.1 has been released:
        https://matplotlib.org/gallery/index.html#statistics

        I update this gist according to the version there, because thanks to the matplotlib community
        the code has improved quite a bit.

        Parameters
        ----------
        x, y : array_like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        Returns
        -------
        matplotlib.patches.Ellipse

        Other parameters
        ----------------
        kwargs : `~matplotlib.patches.Patch` properties
        """
        if x.shape != y.shape:
            raise ValueError("x and y must be the same size")

        ax.scatter(
            x.mean(),
            y.mean(),
            c=color,
            marker="X",
            linewidths=0.6,
            edgecolor="#FFF",
            s=10,
        )
        cov = torch.cov(torch.stack((x, y)))
        pearson = cov[0, 1] / torch.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = torch.sqrt(1 + pearson)
        ell_radius_y = torch.sqrt(1 - pearson)
        ellipse = Ellipse(
            (0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor=facecolor,
            edgecolor=color,
            alpha=alpha,
            **kwargs
        )

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = torch.sqrt(cov[0, 0]) * n_std
        mean_x = torch.mean(x)

        # calculating the stdandard deviation of y ...
        scale_y = torch.sqrt(cov[1, 1]) * n_std
        mean_y = torch.mean(y)

        transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(mean_x, mean_y)
        )

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
        # render plot with "plt.show()".

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
                label="Difference",
            )
            ax2.set_ylabel("Difference", color="#d6272880")
            ax2.tick_params(axis="y", labelcolor="#d6272880", color="#d6272880")
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

    @staticmethod
    def is_out_of_lim(
        torch_list: List[torch.Tensor], lims: Tuple[float], coordinate_idx=0
    ):
        is_out_of_min = True in [
            element[:, :, coordinate_idx].mean() < lims[0] for element in torch_list
        ]
        is_out_of_max = True in [
            element[:, :, coordinate_idx].mean() > lims[1] for element in torch_list
        ]
        return is_out_of_min | is_out_of_max

    @staticmethod
    def mean(torch_list: List[torch.Tensor], coordinate_idx: int = 0):
        return (
            torch.cat([element[:, :, coordinate_idx] for element in torch_list], dim=1)
            .mean()
            .item()
        )

    @staticmethod
    def switch_lims_if_out_of_lim(torch_list, lims_x, lims_y):
        if Plot.is_out_of_lim(torch_list, lims_x, 0):
            mean_x = Plot.mean(torch_list, 0)
            lims_x = (
                mean_x - (lims_x[1] - lims_x[0]) / 2,
                mean_x + (lims_x[1] - lims_x[0]) / 2,
            )
        if Plot.is_out_of_lim(torch_list, lims_y, 1):
            mean_y = Plot.mean(torch_list, 1)
            lims_y = (
                mean_y - (lims_y[1] - lims_y[0]) / 2,
                mean_y + (lims_y[1] - lims_y[0]) / 2,
            )
        return lims_x, lims_y
