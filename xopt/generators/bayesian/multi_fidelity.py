import logging
from copy import deepcopy
from typing import Callable, Dict, Literal, Optional

import pandas as pd
import torch
from botorch.acquisition import (
    FixedFeatureAcquisitionFunction,
    GenericMCObjective,
    qUpperConfidenceBound,
)
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from pydantic import Field, field_validator

from xopt.generators.bayesian.custom_botorch.constrained_acquisition import (
    ConstrainedMCAcquisitionFunction,
)
from xopt.generators.bayesian.custom_botorch.multi_fidelity import NMOMF
from xopt.generators.bayesian.mobo import MOBOGenerator
from xopt.vocs import ObjectiveEnum, VOCS
from xopt.numerical_optimizer import LBFGSMixedOptimizer

logger = logging.getLogger()


class MultiFidelityGenerator(MOBOGenerator):
    """
    Implements Multi-fidelity Bayesian optimization.

    Attributes
    ----------
    name : str
        The name of the generator.
    fidelity_parameter : Literal["s"]
        The fidelity parameter name.
    cost_function : Callable
        Callable function that describes the cost of evaluating the objective function.
    reference_point : Optional[Dict[str, float]]
        The reference point for multi-objective optimization.
    supports_multi_objective : bool
        Indicates if the generator supports multi-objective optimization.
    supports_batch_generation : bool
        Indicates if the generator supports batch candidate generation.

    Methods
    -------
    validate_vocs(cls, v: VOCS) -> VOCS
        Validate the VOCS for the generator.
    calculate_total_cost(self, data: pd.DataFrame = None) -> float
        Calculate the total cost of data samples using the fidelity parameter.
    get_acquisition(self, model: torch.nn.Module) -> NMOMF
        Get the acquisition function for Bayesian Optimization.
    _get_acquisition(self, model: torch.nn.Module) -> NMOMF
        Create the Multi-Fidelity Knowledge Gradient acquisition function.
    add_data(self, new_data: pd.DataFrame)
        Add new data to the generator.
    fidelity_variable_index(self) -> int
        Get the index of the fidelity variable.
    fidelity_objective_index(self) -> int
        Get the index of the fidelity objective.
    get_optimum(self) -> pd.DataFrame
        Select the best point at the maximum fidelity.
    """

    name = "multi_fidelity"
    fidelity_parameter: Literal["s"] = Field(
        "s", description="fidelity parameter name", exclude=True
    )
    cost_function: Callable = Field(
        lambda x: x + 1.0,
        description="callable function that describes the cost "
        "of evaluating the objective function",
        exclude=True,
    )
    reference_point: Optional[Dict[str, float]] = None
    supports_multi_objective: bool = True
    supports_batch_generation: bool = True

    __doc__ = """Implements Multi-fidelity Bayesian optimization
        Assumes a fidelity parameter [0,1]
        """

    @field_validator("vocs", mode="before")
    def validate_vocs(cls, v: VOCS) -> VOCS:
        """
        Validate the VOCS for the generator.

        Parameters
        ----------
        v : VOCS
            The VOCS to be validated.

        Returns
        -------
        VOCS
            The validated VOCS.

        Raises
        ------
        ValueError
            If constraints are present in the VOCS.
        """
        v.variables["s"] = [0, 1]
        v.objectives["s"] = ObjectiveEnum("MAXIMIZE")
        if len(v.constraints):
            raise ValueError(
                "constraints are not currently supported in multi-fidelity BO"
            )

        return v

    def __init__(self, **kwargs):
        reference_point = kwargs.pop("reference_point", None)
        vocs = kwargs.get("vocs")
        # set reference point
        if reference_point is None:
            reference_point = {}
            for name, val in vocs.objectives.items():
                if name != "s":
                    if val == "MAXIMIZE":
                        reference_point.update({name: -100.0})
                    else:
                        reference_point.update({name: 100.0})

        reference_point.update({"s": 0.0})

        super(MultiFidelityGenerator, self).__init__(
            **kwargs, reference_point=reference_point
        )

    def calculate_total_cost(self, data: pd.DataFrame = None) -> float:
        """
        Calculate the total cost of data samples using the fidelity parameter.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The data samples, by default None.

        Returns
        -------
        float
            The total cost of the data samples.
        """
        if data is None:
            data = self.data

        f_data = self.get_input_data(data)

        # apply callable function to get costs
        return self.cost_function(f_data[..., self.fidelity_variable_index]).sum()

    def get_acquisition(self, model: torch.nn.Module) -> NMOMF:
        """
        Get the acquisition function for Bayesian Optimization.

        Parameters
        ----------
        model : torch.nn.Module
            The model used for Bayesian Optimization.

        Returns
        -------
        NMOMF
            The acquisition function.
        """
        if model is None:
            raise ValueError("model cannot be None")

        # get base acquisition function
        acq = self._get_acquisition(model)
        return acq

    def _get_acquisition(self, model: torch.nn.Module) -> NMOMF:
        """
        Create the Multi-Fidelity Knowledge Gradient acquisition function.

        In order for MFKG to evaluate the information gain, it uses the model to
        predict the function value at the highest fidelity after conditioning
        on the observation. This is handled by the project argument, which specifies
        how to transform a tensor X to its target fidelity. We use a default helper
        function called project_to_target_fidelity to achieve this.

        An important point to keep in mind: in the case of standard KG, one can ignore
        the current value and simply optimize the expected maximum posterior mean of the
        next stage. However, for MFKG, since the goal is optimize information gain per
        cost, it is important to first compute the current value (i.e., maximum of the
        posterior mean at the target fidelity). To accomplish this, we use a
        FixedFeatureAcquisitionFunction on top of a PosteriorMean.

        Parameters
        ----------
        model : torch.nn.Module
            The model used for Bayesian Optimization.

        Returns
        -------
        NMOMF
            The Multi-Fidelity Knowledge Gradient acquisition function.
        """
        X_baseline = self.get_input_data(self.data)

        # wrap the cost function such that it only has to accept the fidelity parameter
        def true_cost_function(X: torch.Tensor) -> torch.Tensor:
            return self.cost_function(X[..., self.fidelity_variable_index])

        acq_func = NMOMF(
            model=model,
            X_baseline=X_baseline,
            ref_point=self.torch_reference_point,
            cost_call=true_cost_function,
            objective=self._get_objective(),
            constraints=self._get_constraint_callables(),
            cache_root=False,
            prune_baseline=True,
        )

        return acq_func

    def add_data(self, new_data: pd.DataFrame):
        """
        Add new data to the generator.

        Parameters
        ----------
        new_data : pd.DataFrame
            The new data to be added.

        Raises
        ------
        ValueError
            If the fidelity parameter is not in the new data or if the fidelity
            values are outside the range [0,1].
        """
        if self.fidelity_parameter not in new_data:
            raise ValueError(
                f"fidelity parameter {self.fidelity_parameter} must be in added data"
            )

        # overwrite add data to check for valid fidelity values
        if (new_data[self.fidelity_parameter] > 1.0).any() or (
            new_data[self.fidelity_parameter] < 0.0
        ).any():
            raise ValueError("cannot add fidelity data that is outside the range [0,1]")
        super().add_data(new_data)

    @property
    def fidelity_variable_index(self) -> int:
        """
        Get the index of the fidelity variable.

        Returns
        -------
        int
            The index of the fidelity variable.
        """
        return self.vocs.variable_names.index(self.fidelity_parameter)

    @property
    def fidelity_objective_index(self) -> int:
        """
        Get the index of the fidelity objective.

        Returns
        -------
        int
            The index of the fidelity objective.
        """
        return self.vocs.objective_names.index(self.fidelity_parameter)

    def get_optimum(self) -> pd.DataFrame:
        """
        Select the best point at the maximum fidelity.

        Returns
        -------
        pd.DataFrame
            The best point at the maximum fidelity.
        """
        # define single objective based on vocs
        weights = torch.zeros(self.vocs.n_outputs, **self.tkwargs)
        for idx, ele in enumerate(self.vocs.objective_names):
            if self.vocs.objectives[ele] == "MINIMIZE":
                weights[idx] = -1.0
            elif self.vocs.objectives[ele] == "MAXIMIZE":
                weights[idx] = 1.0

        def obj_callable(
            Z: torch.Tensor, X: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            return torch.matmul(Z, weights.reshape(-1, 1)).squeeze(-1)

        c_posterior_mean = ConstrainedMCAcquisitionFunction(
            self.model,
            qUpperConfidenceBound(
                model=self.model, beta=0.0, objective=GenericMCObjective(obj_callable)
            ),
            self._get_constraint_callables(),
        )

        max_fidelity_c_posterior_mean = FixedFeatureAcquisitionFunction(
            c_posterior_mean,
            self.vocs.n_variables,
            [self.fidelity_variable_index],
            [1.0],
        )

        boundst = self._get_bounds().T
        fixed_bounds = torch.cat(
            (
                boundst[: self.fidelity_variable_index],
                boundst[self.fidelity_variable_index + 1 :],
            )
        ).T

        result = self.numerical_optimizer.optimize(
            max_fidelity_c_posterior_mean, fixed_bounds, 1
        )

        vnames = deepcopy(self.vocs.variable_names)
        del vnames[self.fidelity_variable_index]
        df = pd.DataFrame(result.detach().cpu().numpy(), columns=vnames)
        df[self.fidelity_parameter] = 1.0

        return self.vocs.convert_dataframe_to_inputs(df)


class MultiFidelityDiscreteGenerator(MultiFidelityGenerator):
    """
    Multi-fidelity Bayesian generator that supports discrete variables.
    Inherits from MultiFidelityGenerator and adds support for discrete variables.
    Uses the propose_candidates method from BayesianGenerator.
    """

    supports_discrete_variables: bool = True

    # Explicitly override the default numerical_optimizer for this class with type annotation for Pydantic
    numerical_optimizer: LBFGSMixedOptimizer = LBFGSMixedOptimizer()

    def _get_discrete_variables_list(self) -> list[dict[int, float]]:
        """
        Get a list of fixed features for discrete variables.

        Returns
        -------
        list of dict
            A list of dictionaries specifying fixed features index and corresponding feature values.
        """
        discrete_variable_list = []
        discrete_vars = self.vocs.get_discrete_variable_indices()
        if discrete_vars:
            for name, index in discrete_vars.items():
                discrete_vals = self.vocs.discrete_variables[name]
                for val in discrete_vals:
                    discrete_variable_list.append({index: val})
        else:
            discrete_variable_list.append({})  # No discrete variables, empty dict

        return discrete_variable_list

    def propose_candidates(self, model, n_candidates: int = 1):
        # modified existing code from BayesianGenerator.propose_candidates
        if self.turbo_controller is not None:
            self.turbo_controller.update_state(self, n_candidates)

        # calculate optimization bounds
        bounds = self._get_optimization_bounds()

        # get list of fixed features and allowed values
        discrete_variables_list = self._get_discrete_variables_list()

        # get acquisition function
        acq_funct = self.get_acquisition(model)

        # get initial candidates to start acquisition function optimization
        initial_points = self._get_initial_conditions(n_candidates)

        # get candidates
        if isinstance(self.numerical_optimizer, LBFGSMixedOptimizer):
            candidates = self.numerical_optimizer.optimize(
                acq_funct,
                bounds,
                discrete_variables_list,
                n_candidates,
                batch_initial_conditions=initial_points,
            )

        return candidates


class MFHVKGWrapped(qMultiFidelityHypervolumeKnowledgeGradient):
    """
    Wrapper around qMultiFidelityHypervolumeKnowledgeGradient that
    makes it behave t-batch-wise for Xopt / optimize_acqf_mixed.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape (t_batch, q, d).
    Returns
    -------
    torch.Tensor
        quantifies the acquisition value for each t-batch.

    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Handle the common case: X has a t-batch dimension
        if X.dim() == 3:
            t_batch = X.shape[0]
            vals = []

            # Call the *undecorated* base forward once per t-batch row
            base_forward = (
                qMultiFidelityHypervolumeKnowledgeGradient.forward.__wrapped__
            )

            for i in range(t_batch):
                Xi = X[i : i + 1]  # shape (1, q, d)
                vi = base_forward(self, Xi)  # shape (1,) from MF-HVKG
                vals.append(vi.squeeze(-1))  # -> scalar

            return torch.stack(vals, dim=0)  # (t_batch,)

        # Fallback: no t-batch, just call the undecorated base
        base_forward = qMultiFidelityHypervolumeKnowledgeGradient.forward.__wrapped__
        return base_forward(self, X)


class MultiFidelityHVKGDiscreteGenerator(MultiFidelityDiscreteGenerator):
    """
    Multi-fidelity Bayesian generator using Multi-Fidelity Hypervolume Knowledge Gradient (MF-HVKG)
    with discrete variable support.

    - Standard MF-HVKG by default (no cost awareness).
    - Becomes cost-aware only if `cost_function` is explicitly provided.
    """

    # Override inherited default so MF-HVKG is standard unless user opts in
    cost_function: Optional[Callable] = None

    # Explicitly override the default numerical_optimizer for this class
    numerical_optimizer: LBFGSMixedOptimizer = LBFGSMixedOptimizer()

    def build_model(
        self,
        input_names,
        outcome_names,
        data,
        input_bounds=None,
        dtype: torch.dtype = torch.double,
        device: str = "cpu",
    ):
        """
        Build and return a SingleTaskMultiFidelityGP model for multi-fidelity optimization.
        """
        X = torch.tensor(data[input_names].values, dtype=dtype, device=device)
        Y = torch.tensor(data[outcome_names].values, dtype=dtype, device=device)

        fidelity_features = [self.fidelity_variable_index]

        return SingleTaskMultiFidelityGP(
            X,
            Y,
            fidelity_features=fidelity_features,
        )

    def get_acquisition(self, model: torch.nn.Module):
        """
        Get the Multi-Fidelity Knowledge Gradient acquisition function for Bayesian Optimization.
        """
        if model is None:
            raise ValueError("model cannot be None")
        return self._get_acquisition(model)

    def _compute_current_value(self, model: torch.nn.Module) -> torch.Tensor:
        """Helper to get the hypervolume of the current hypervolume
        maximizing set.
        """

        fid_idx = self.fidelity_variable_index
        max_fidelity = self.vocs.variables[self.fidelity_parameter][-1]

        # create helper function for hypervolume calculation at max fidelity
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=_get_hv_value_function(
                model=model,
                ref_point=self.torch_reference_point,
                sampler=SobolQMCNormalSampler(
                    sample_shape=torch.Size([32]),
                ),
                use_posterior_mean=True,
            ),
            d=self.vocs.n_variables,
            columns=[fid_idx],
            values=[max_fidelity],
        )

        # Get bounds and remove the fidelity dimension
        #    self._get_bounds() returns a (2, d) tensor
        bounds = self._get_bounds()
        boundst = bounds.T  # (d, 2)
        bounds_no_fid = torch.cat(
            (boundst[:fid_idx], boundst[fid_idx + 1 :]), dim=0
        ).T  # (2, d-1)

        # Optimize max fidelity hypervolume over remaining variables
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=bounds_no_fid,
            q=1,
            num_restarts=10,
            raw_samples=256,
            return_best_only=True,
            options={
                "nonnegative": True,
                "maxiter": 200,
            },
        )

        # current_value has shape (1,); squeeze to scalar
        return current_value.squeeze()

    def _get_acquisition(self, model: torch.nn.Module):
        """
        Create the Multi-Fidelity Knowledge Gradient acquisition function.

        Steps:
        - Compute the current best value at the highest fidelity (by optimizing posterior mean).
        - Build a projection to the target fidelity (max fidelity from VOCS).
        - Wrap cost_function only if explicitly provided.
        - Construct MFKGWrapped (MF-KG with proper BoTorch transform).
        """
        fid_idx = self.fidelity_variable_index

        # Compute current best at target fidelity via optimization
        current_value = self._compute_current_value(model)

        # Build project() function to map any X to max fidelity
        max_fidelity = self.vocs.variables[self.fidelity_parameter][-1]

        def project_to_target_fidelity(X: torch.Tensor) -> torch.Tensor:
            X_proj = X.clone()
            X_proj[..., fid_idx] = max_fidelity
            return X_proj

        # Build cost-aware utility only if cost_function is provided
        if self.cost_function is None:
            cost_utility = None
        else:

            def cost_utility(
                X: torch.Tensor, deltas: torch.Tensor = None, **kwargs
            ) -> torch.Tensor:
                # cost depends only on fidelity coordinate
                # return self.cost_function(X[..., fid_idx])
                s = X[:, 0, fid_idx]  # shape (t_batch,)
                return self.cost_function(s).unsqueeze(-1)

        # Construct MF-HVKG acquisition function
        acq_func = MFHVKGWrapped(
            model=model,
            ref_point=self.torch_reference_point,
            num_fantasies=2,
            num_pareto=1,
            current_value=current_value,
            cost_aware_utility=cost_utility,
            target_fidelities={fid_idx: max_fidelity},
            project=project_to_target_fidelity,
        )

        return acq_func
