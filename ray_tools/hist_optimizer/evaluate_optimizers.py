import sys
sys.path.append("../..")
from ray_tools.hist_optimizer.hist_optimizer import *
from ray_nn.nn.xy_hist_data_models import HistSurrogateEngine, Model, StandardizeXYHist
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def eval_optimizer_iterative(optimize_dict, model, repetitions=10):
    result_dict = {}

    for key, (optimize_fn, param_grid) in optimize_dict.items():
        print(f"Running iterative tuning for: {key}")

        # Sweep only parameters that include a "values" list
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
                run_results = []
                run_progresses = []

                for i in range(repetitions):
                    offsets_selected, uncompensated_parameters_selected, compensated_parameters_selected = find_good_offset_problem(model, fixed_parameters = [8, 14, 20, 21, 27, 28, 34], seed=i)
        
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
    for optimizer_name, param_results in result_dict.items():
        _, param_grid = optimize_dict[optimizer_name]

        for param_name, param_values in param_results.items():
            plt.figure(figsize=(8, 3))

            # Get LaTeX label from nested dict, fallback to param_name
            display_name = param_grid[param_name].get("label", param_name)

            all_means = []

            for param_value_str, data in param_values.items():
                progresses = data['progresses'].cpu()
                mean_progress = progresses.mean(dim=0)
                std_progress = progresses.std(dim=0)

                all_means.append(mean_progress)

                x = range(len(mean_progress))

                # Plot mean line
                plt.plot(x, mean_progress, label=f"{display_name}={param_value_str}", alpha=0.8)
                # Plot ± std shading
                plt.fill_between(
                    x,
                    mean_progress - std_progress,
                    mean_progress + std_progress,
                    alpha=0.2
                )

            plt.tick_params(axis='both', which='major', labelsize=11)
            plt.xlabel("Iteration [#]")
            plt.ylabel(r"$\mathcal{L}_h(\mathbf{x})$")

            # Set log scale AFTER plotting
            if "scale" in param_grid[param_name]:
                plt.yscale(param_grid[param_name]["scale"])

            # Set ymin only based on mean_progress (ignore std)
            all_means_tensor = torch.stack(all_means)
            ymin = all_means_tensor.min().item()

            if "scale" in param_grid[param_name] and param_grid[param_name]["scale"] == "log":
                ymin = max(ymin*0.8, 1e-8)  # avoid log(0)
                plt.ylim(bottom=ymin)

            

            plt.legend(fontsize=11)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{optimizer_name}_{param_name}.pdf")
            plt.show()

if __name__ == "__main__":
    file_root = '../../'
    model_path = os.path.join(file_root, "outputs/xy_hist/s021yw7n/checkpoints/epoch=235-step=70000000.ckpt")
    surrogate_engine = HistSurrogateEngine(checkpoint_path=model_path)
    model = Model(path=model_path)

    optimize_dict = {
        "SA": (optimize_sa, {
            "step_size": {
                "values": [0.01, 0.1, 0.2, 0.5, 1.0],
                "label": r"$\eta$"
            },
            "T_start": {
                "values": [0.01, 0.1, 10.0, 100.0, 1000.0],
                "label": r"$k_0$"
            },
            "alpha": {
                "values": [0.001, 0.005, 0.01, 0.5, 0.99],
                "label": r"$\alpha$"
            },
        }),
    }
    
    result_dict = eval_optimizer_iterative(optimize_dict, model, repetitions=10)
    plot_result_dict(result_dict, optimize_dict)
    
    optimize_dict = {
        "GA": (optimize_evotorch_ga, {
            "num_candidates": {
                "values": [10, 100, 200, 500],
                "label": r"$p$",
                 "scale": "log",
            },
             "crossover_points": {
                 "values": [1, 5, 10, 15, 20],
                 "label": r"$k_c$",
                 "scale": "log",
             },
             "tournament_size": {
                 "values": [1, 3, 5, 10, 15, 20],
                 "label": r"$k_t$",
                 "scale": "log",
             },
             "mutation_rate": {
                 "values": [0.001, 0.01, 0.05, 0.1, 0.2],
                 "label": r"$r_m$",
                 "scale": "log",
             },
             "mutation_scale": {
                 "values": [0.001, 0.01, 0.05, 0.1, 0.2],
                 "label": r"$s_m$",
                 "scale": "log",
             },
             "crossover_rate": {
                 "values": [0.1, 0.3, 0.5, 0.8, 0.9],
                 "label": r"$r_c$",
                 "scale": "log",
             },
        }),
    }

    result_dict = eval_optimizer_iterative(optimize_dict, model, repetitions=10)
    plot_result_dict(result_dict, optimize_dict)
    
    optimize_dict = {
        "GD": (optimize_gd, {
             "learning_rate": {
                 "values": [1e-7, 1e-6, 0.0001, 0.001, 0.01, 0.1],
                 "label": r"$\eta$",
                 "scale": "linear",
             },
        }),
    }
    
    result_dict = eval_optimizer_iterative(optimize_dict, model, repetitions=10)
    plot_result_dict(result_dict, optimize_dict)
