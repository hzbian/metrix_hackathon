import os
import sys
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.append("../..")
from ray_tools.hist_optimizer.hist_optimizer import *
from ray_nn.nn.xy_hist_data_models import HistSurrogateEngine, Model, StandardizeXYHist
from matplotlib.ticker import FuncFormatter
from matplotlib.pylab import cycler

def space_thousands(x, pos):
    return f"{int(x):,}".replace(",", "\u202f")


# -----------------------------
# Core functions
# -----------------------------
def eval_optimizer_iterative(optimize_dict, model, repetitions=10):
    result_dict = {}

    for key, (optimize_fn, param_grid) in optimize_dict.items():
        print(f"Running iterative tuning for: {key}")

        sweep_params = {
            k: v for k, v in param_grid.items()
            if isinstance(v, dict) and isinstance(v.get("values"), list)
        }

        result_dict[key] = {}

        for param_name, param_info in sweep_params.items():
            param_values = param_info["values"]
            result_dict[key][param_name] = {}

            print(f"\n{key} Sweeping parameter: {param_name}")

            for val in tqdm(param_values, desc=f"{param_name}", leave=False):
                run_results, run_progresses = [], []

                for i in range(repetitions):
                    offsets_selected, uncompensated_parameters_selected, compensated_parameters_selected = find_good_offset_problem(
                        model,
                        fixed_parameters=[8, 14, 20, 21, 27, 28, 34],
                        seed=i + 10000000
                    )

                    with torch.no_grad():
                        observed_rays = model(compensated_parameters_selected)

                    _, result, progress = optimize_fn(
                        model,
                        observed_rays,
                        uncompensated_parameters_selected,
                        **{param_name: val},
                        seed=i
                    )

                    run_results.append(result)
                    run_progresses.append(progress)

                mean_result = torch.tensor(run_results).mean()
                print(f"{param_name} = {val} → Mean result: {mean_result:.10f}")

                result_dict[key][param_name][f"{val}"] = {
                    "mean_result": mean_result,
                    "progresses": torch.tensor(run_progresses)
                }

    return result_dict


def plot_result_dict(result_dict, optimize_dict):
    os.makedirs("outputs", exist_ok=True)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    custom_colors = default_colors[:2] + default_colors[3:]
    
    for optimizer_name, param_results in result_dict.items():
        _, param_grid = optimize_dict[optimizer_name]

        for param_name, param_values in param_results.items():
            plt.figure(figsize=(8, 3))
            plt.rcParams['axes.prop_cycle'] = cycler(color=custom_colors)
            display_name = param_grid[param_name].get("label", param_name)
            all_means = []

            for param_value_str, data in param_values.items():
                progresses = data["progresses"].cpu()
                mean_progress = progresses.mean(dim=0)
                std_progress = progresses.std(dim=0)
                all_means.append(mean_progress)
                x = range(len(mean_progress))
                plt.plot(x, mean_progress, label=f"{display_name}={param_value_str}", alpha=0.8)
                plt.fill_between(x, mean_progress - std_progress, mean_progress + std_progress, alpha=0.2)
            plt.gca().xaxis.set_major_formatter(FuncFormatter(space_thousands))
            plt.gca().yaxis.set_major_formatter(FuncFormatter(space_thousands))
            plt.tick_params(axis="both", which="major", labelsize=11)
            plt.xlabel("Iteration [#]")
            plt.ylabel(r"$\mathcal{L}_h(\mathbf{x})$")

            if "scale" in param_grid[param_name]:
                plt.yscale(param_grid[param_name]["scale"])

            all_means_tensor = torch.stack(all_means)
            ymin = all_means_tensor.min().item()
            if "scale" in param_grid[param_name] and param_grid[param_name]["scale"] == "log":
                ymin = max(ymin * 0.8, 1e-8)
                plt.ylim(bottom=ymin)

            loc = param_grid[param_name].get("loc", "best")
            plt.legend(fontsize=11, loc=loc)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"outputs/{optimizer_name}_{param_name}.pdf")
            plt.close()


# -----------------------------
# Build sweep list
# -----------------------------
def build_sweep_list(optimize_dict):
    """Flatten optimize_dict → list of (optimizer_name, optimize_fn, param_name, param_dict)"""
    sweeps = []
    for opt_name, (opt_fn, param_grid) in optimize_dict.items():
        for param_name, param_info in param_grid.items():
            if isinstance(param_info, dict) and isinstance(param_info.get("values"), list):
                sweeps.append((opt_name, opt_fn, param_name, param_grid))
    return sweeps


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int, help="Sweep index (0–N)")
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    # Load model
    file_root = ''
    model_path = os.path.join(file_root, "outputs/xy_hist/s021yw7n/checkpoints/epoch=235-step=70000000.ckpt")
    surrogate_engine = HistSurrogateEngine(checkpoint_path=model_path)
    model = Model(path=model_path)

    # Full optimize_dict
    optimize_dict = {
        "BLOP": (optimize_blop, {
            "acq": {"values": ["ei", "lcb"], "label": r"$a$", "scale": "log"},
            "warm_up_iterations": {"values": [16, 32, 64], "label": r"$l_\mathrm{warm}$", "scale": "log"},
            "transform": {"values": [None, "log"], "label": r"$t$", "scale": "log"},
            "ucb_beta": {"values": [0.4, 1.0, 5.0, 10.0, 15.0], "label": r"$\beta$", "scale": "log"},
            "empty_image_threshold": {"values": [1e-10, 1e-5, 1e-4, 1e-3], "label": r"$\theta$", "scale": "log"},
        }),
        "SA": (optimize_sa, {
            "step_size": {"values": [0.001, 0.01, 0.1, 0.2, 0.5], "label": r"$\eta$", "loc": "lower left", "scale": "log"},
            "T_start": {"values": [1e-6,1e-5, 1e-4, 1e-3], "label": r"$t_\mathrm{start}$", "scale": "log"},
            "cooling_schedule": {"values": ["linear", "exp"], "label": r"$t$", "scale": "log"},
        }),
        "GD": (optimize_gd, {
            "learning_rate": {"values": [0.0001, 0.001, 0.01, 0.1], "label": r"$\eta$", "scale": "log", "loc": "upper right"},
        }),
        "GA": (optimize_evotorch_ga, {
            "num_candidates": {"values": [10, 100, 200, 500], "label": r"$p$", "scale": "log"},
            "tournament_size": {"values": [1, 3, 5, 10, 15, 20], "label": r"$k_t$", "scale": "log"},
            "mutation_rate": {"values": [0.001, 0.01, 0.05, 0.1, 0.2], "label": r"$r_m$", "scale": "log"},
            "mutation_scale": {"values": [0.001, 0.01, 0.05, 0.1, 0.2], "label": r"$s_m$", "scale": "log"},
            "sbx_eta": {"values": [1, 5, 10, 50, 100], "label": r"$\eta$", "scale": "log", "loc": "lower left"},
            "sbx_crossover_rate": {"values": [0.1, 0.3, 0.5, 0.8, 0.9], "label": r"$r_c$", "scale": "log", "loc": "lower left"},
        }),
    }

    # Flatten all sweeps
    sweeps = build_sweep_list(optimize_dict)
    total = len(sweeps)

    if args.index < 0 or args.index >= total:
        raise ValueError(f"Invalid index {args.index}. Must be between 0 and {total-1}")

    opt_name, opt_fn, param_name, param_grid = sweeps[args.index]
    print(f"Running sweep {args.index+1}/{total}: {opt_name}.{param_name}")

    sub_optimize_dict = {opt_name: (opt_fn, {param_name: param_grid[param_name]})}

    # Run
    result_dict = eval_optimizer_iterative(sub_optimize_dict, model, repetitions=args.reps)
    plot_result_dict(result_dict, sub_optimize_dict)

    # Save per-sweep results
    os.makedirs("outputs", exist_ok=True)
    torch.save(result_dict, f"outputs/result_{opt_name}_{param_name}.pt")

    print(f"Finished {opt_name}.{param_name}")

