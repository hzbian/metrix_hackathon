
from collections.abc import Callable
from typing import Any
import random
import math
from tqdm import tqdm
from ray_optim.optimizer_backend.base import OptimizerBackend
from ray_optim.target import Target
from ray_tools.base.parameter import MutableParameter, NumericalParameter, RayParameterContainer
import os
import math
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state



class OptimizerBackendTurbo(OptimizerBackend):
    def __init__(self, batch_size=4):
        self.batch_size = batch_size
        return
    @staticmethod 
    def get_initial_points(dim, n_pts, seed=0):
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
        return X_init
    @staticmethod
    def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    ):
        assert acqf in ("ts", "ei")
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if acqf == "ts":
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=batch_size)

        elif acqf == "ei":
            ei = qExpectedImprovement(model, Y.max())
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        return X_next
    
    def torch_objective(self, objective, target: Target):
        def output_objective(input: torch.Tensor):
            optimize_parameters = target.search_space.clone()
            mutable_index = 0
            optimize_parameters_list = []
            for entry in input:
                for key, value in optimize_parameters.items():
                    if isinstance(value, MutableParameter):
                        optimize_parameters[key] = NumericalParameter(
                            entry[mutable_index].item()
                        )
                        mutable_index += 1
                optimize_parameters_list.append(optimize_parameters)
            output = objective(optimize_parameters_list, target=target)
            return torch.tensor(output, dtype=input.dtype, device=input.device)

        return output_objective


    def optimize(self, objective: Callable[[list[RayParameterContainer], Target], list[float]], iterations: int, target: Target, starting_point: dict[str, float] | None = None) -> tuple[dict[str, float], dict[str, float]]:
        optimize_parameters: RayParameterContainer = target.search_space.clone()
        dim = len([key for key, value in optimize_parameters.items() if isinstance(value, MutableParameter)])
        n_init = 2*dim
        max_cholesky_size = float("inf")
        self.state = TurboState(dim=dim, batch_size=self.batch_size)
        X_turbo = OptimizerBackendTurbo.get_initial_points(dim, n_init)
        def eval_objective(x):
            return self.torch_objective(objective, target)(x.unsqueeze(dim=0)) # here we have an unsqueeze since we have no batch for now?
        
        Y_turbo = torch.tensor(
            [eval_objective(x) for x in X_turbo], dtype=dtype, device=device
        ).unsqueeze(-1)

        state = TurboState(dim, batch_size=self.batch_size)

        NUM_RESTARTS = 10 
        RAW_SAMPLES = 512
        N_CANDIDATES = min(5000, max(2000, 200 * dim))

        torch.manual_seed(0)

        while not state.restart_triggered:  # Run until TuRBO converges
            # Fit a GP model
            train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(
                    nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
                )
            )
            model =SingleTaskGP(
                X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # Do the fitting and acquisition function optimization inside the Cholesky context
            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                # Fit the model
                fit_gpytorch_mll(mll)

                # Create a batch
                X_next = OptimizerBackendTurbo.generate_batch(
                    state=state,
                    model=model,
                    X=X_turbo,
                    Y=train_Y,
                    batch_size=self.batch_size,
                    n_candidates=N_CANDIDATES,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                    acqf="ts",
                )

            Y_next = torch.tensor(
                [eval_objective(x) for x in X_next], dtype=dtype, device=device
            ).unsqueeze(-1)

            # Update state
            state = update_state(state=state, Y_next=Y_next)

            # Append data
            X_turbo = torch.cat((X_turbo, X_next), dim=0)
            Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

            # Print current status
            print(
                f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
            )


        return {},{}
        optimize_parameters: RayParameterContainer = target.search_space.clone()
        if starting_point is not None:
            for key in starting_point.keys():
                current_parameter = optimize_parameters[key]
                if isinstance(current_parameter, NumericalParameter):
                    current_parameter.value = starting_point[key]
        mutable_parameters_keys = [key for key, value in optimize_parameters.items() if isinstance(value, MutableParameter)] 
        current_parameters = optimize_parameters
        all_time_best_loss: float = objective([current_parameters], target)[0]
        all_time_best_parameters: RayParameterContainer = current_parameters
        current_best_loss: float = all_time_best_loss
        q = tqdm(range(iterations))
        for i in q:
            q.set_postfix({"best_loss": all_time_best_loss, "cur_loss": current_best_loss})
            perturbation_key = random.choice(mutable_parameters_keys)
            perturbation_sign = 1 if random.random() < 0.5 else -1
            perturbation_parameters = current_parameters[perturbation_key]
            assert isinstance(perturbation_parameters, MutableParameter)
            if self.annealing:
                annealing_factor = math.cos(i/(iterations/(5*math.pi)))+1.
            else:
                annealing_factor = 1.
            perturbation_value: float = perturbation_sign * (perturbation_parameters.value_lims[1]-perturbation_parameters.value_lims[0]) / 2 * self.step_size * annealing_factor
            current_parameters_copy: RayParameterContainer = current_parameters.clone()
            current_parameters_copy.perturb(RayParameterContainer({perturbation_key: NumericalParameter(perturbation_value)}))
            losses: list[float] = objective([current_parameters_copy], target)
            current_best_loss = losses[0]
            if current_best_loss < all_time_best_loss:
                current_parameters = current_parameters_copy
                all_time_best_loss = current_best_loss
                all_time_best_parameters = current_parameters
        return all_time_best_parameters.to_value_dict(), {"loss": all_time_best_loss}