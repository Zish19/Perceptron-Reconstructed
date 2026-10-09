"""Paper-inspired photoperceptron architecture and reinforcement rules.

The implementation models the following stages:

    retinal S-points -> projection A-units -> association A-units -> response R-units

The S-to-A connections have excitatory (+1) and inhibitory (-1) signs. Each A-unit
uses an all-or-none fixed-threshold response. Response units compete by a winner-
take-all rule. A-unit values change during reinforcement.

This is an explicit, tractable implementation of the paper's core organization,
not a line-by-line numerical reproduction of every model variant and derivation
in the 1958 article. See docs/model_spec.md for assumptions and deviations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int16]
BoolArray = NDArray[np.bool_]
LearningRule = Literal["alpha", "beta", "gamma", "bivalent_gamma"]
DiscriminationMode = Literal["sum", "mean"]
ProjectionLocality = Literal["local", "random"]


@dataclass(frozen=True)
class PerceptronConfig:
    """Configuration for the historical-model simulator.

    Counts are kept small enough for interactive experiments. Excitatory and
    inhibitory origin counts are sampled without replacement for each receiving
    unit. "local" projection connectivity samples origin points near a randomly
    assigned retinal center; "random" uses the whole retina.
    """

    retina_size: int = 16
    n_projection_units: int = 96
    n_association_units: int = 256
    n_responses: int = 2
    excitatory_origins: int = 12
    inhibitory_origins: int = 4
    projection_threshold: int = 3
    association_excitatory_origins: int = 16
    association_inhibitory_origins: int = 4
    association_threshold: int = 1
    projection_locality: ProjectionLocality = "local"
    locality_sigma: float = 3.0
    response_connection_probability: float = 0.25
    disjoint_response_sources: bool = True
    learning_rule: LearningRule = "bivalent_gamma"
    discrimination_mode: DiscriminationMode = "sum"
    learning_rate: float = 0.05
    seed: int = 7

    def validate(self) -> None:
        """Raise ValueError for invalid model configurations."""
        if self.retina_size < 4:
            raise ValueError("retina_size must be at least 4")
        if self.n_projection_units < 1 or self.n_association_units < 2:
            raise ValueError("projection/association layer sizes must be positive")
        if self.n_responses < 2:
            raise ValueError("n_responses must be at least 2")
        if self.n_association_units < self.n_responses:
            raise ValueError("n_association_units must be >= n_responses")
        n_sensors = self.retina_size * self.retina_size
        if self.excitatory_origins < 0 or self.inhibitory_origins < 0:
            raise ValueError("origin counts cannot be negative")
        if self.excitatory_origins + self.inhibitory_origins > n_sensors:
            raise ValueError("projection origin counts exceed the number of S-points")
        if self.association_excitatory_origins < 0 or self.association_inhibitory_origins < 0:
            raise ValueError("association origin counts cannot be negative")
        if self.association_excitatory_origins + self.association_inhibitory_origins > self.n_projection_units:
            raise ValueError("association origin counts exceed the projection layer size")
        if self.projection_locality not in ("local", "random"):
            raise ValueError("projection_locality must be 'local' or 'random'")
        if self.locality_sigma <= 0:
            raise ValueError("locality_sigma must be positive")
        if not 0 < self.response_connection_probability <= 1:
            raise ValueError("response_connection_probability must be in (0, 1]")
        if self.learning_rule not in ("alpha", "beta", "gamma", "bivalent_gamma"):
            raise ValueError("unsupported learning_rule")
        if self.discrimination_mode not in ("sum", "mean"):
            raise ValueError("discrimination_mode must be 'sum' or 'mean'")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.learning_rule == "bivalent_gamma" and self.n_responses != 2:
            raise ValueError("bivalent_gamma currently requires exactly two responses")
        if self.learning_rule == "bivalent_gamma" and not self.disjoint_response_sources:
            raise ValueError("bivalent_gamma requires disjoint response source-sets")


@dataclass(frozen=True)
class ActivationTrace:
    """Intermediate activations from one forward pass."""

    projection_net: IntArray
    projection_active: BoolArray
    association_net: IntArray
    association_active: BoolArray
    response_scores: FloatArray
    winner: int


class RosenblattPerceptron:
    """A compact simulation of Rosenblatt's layered photoperceptron."""

    def __init__(self, config: PerceptronConfig | None = None) -> None:
        self.config = config or PerceptronConfig()
        self.config.validate()
        self.rng = np.random.default_rng(self.config.seed)
        self.n_sensors = self.config.retina_size**2

        self.projection_signs = self._make_projection_connections()
        self.association_signs = self._make_association_connections()
        self.response_sources = self._make_response_sources()

        # Historical interpretation: V is an effective value associated with an
        # A-unit. Equal initial values give every unit the same starting value.
        self.association_values: FloatArray = np.ones(
            self.config.n_association_units, dtype=np.float64
        )
        self.training_steps = 0

    def _sample_signed_row(
        self, n_inputs: int, n_excitatory: int, n_inhibitory: int,
        probabilities: FloatArray | None = None,
    ) -> IntArray:
        """Sample fixed counts of excitatory and inhibitory origin points."""
        p = None if probabilities is None else probabilities / probabilities.sum()
        selected = self.rng.choice(n_inputs, size=n_excitatory + n_inhibitory,
                                   replace=False, p=p)
        row = np.zeros(n_inputs, dtype=np.int8)
        if n_excitatory:
            row[selected[:n_excitatory]] = 1
        if n_inhibitory:
            row[selected[n_excitatory:]] = -1
        return row

    def _make_projection_connections(self) -> IntArray:
        cfg = self.config
        n_sensors = cfg.retina_size**2
        coords = np.indices((cfg.retina_size, cfg.retina_size)).reshape(2, -1).T
        signs = np.zeros((cfg.n_projection_units, n_sensors), dtype=np.int8)
        for unit in range(cfg.n_projection_units):
            probabilities = None
            if cfg.projection_locality == "local":
                center = self.rng.uniform(0, cfg.retina_size - 1, size=2)
                squared_distance = ((coords - center) ** 2).sum(axis=1)
                probabilities = np.exp(-squared_distance / (2 * cfg.locality_sigma**2))
                # Underflow is unlikely at these sizes, but normalization must be safe.
                probabilities = np.maximum(probabilities, np.finfo(float).tiny)
            signs[unit] = self._sample_signed_row(
                n_sensors, cfg.excitatory_origins, cfg.inhibitory_origins, probabilities
            )
        return signs

    def _make_association_connections(self) -> IntArray:
        cfg = self.config
        signs = np.zeros((cfg.n_association_units, cfg.n_projection_units), dtype=np.int8)
        for unit in range(cfg.n_association_units):
            signs[unit] = self._sample_signed_row(
                cfg.n_projection_units,
                cfg.association_excitatory_origins,
                cfg.association_inhibitory_origins,
            )
        return signs

    def _make_response_sources(self) -> BoolArray:
        cfg = self.config
        sources = np.zeros((cfg.n_responses, cfg.n_association_units), dtype=np.bool_)
        if cfg.disjoint_response_sources:
            # Assign every association unit to exactly one source-set, ensuring
            # each response has at least one unit.
            assignment = np.arange(cfg.n_association_units) % cfg.n_responses
            self.rng.shuffle(assignment)
            for response in range(cfg.n_responses):
                sources[response] = assignment == response
        else:
            sources = self.rng.random(
                (cfg.n_responses, cfg.n_association_units)
            ) < cfg.response_connection_probability
            # Ensure no response has an empty source-set.
            for response in range(cfg.n_responses):
                if not sources[response].any():
                    sources[response, self.rng.integers(cfg.n_association_units)] = True
        return sources

    def _validate_stimulus(self, stimulus: NDArray[np.generic]) -> BoolArray:
        array = np.asarray(stimulus)
        if array.size != self.n_sensors:
            raise ValueError(
                f"stimulus has {array.size} values; expected {self.n_sensors} "
                f"({self.config.retina_size}x{self.config.retina_size} retina)"
            )
        if not np.isin(array, [0, 1, False, True]).all():
            raise ValueError("stimulus must contain only binary values (0 or 1)")
        return array.astype(np.bool_, copy=False).reshape(-1)

    def forward(self, stimulus: NDArray[np.generic]) -> ActivationTrace:
        """Propagate one binary stimulus through S, A_I, A_II, and R units."""
        s_points = self._validate_stimulus(stimulus).astype(np.int16)
        projection_net = (self.projection_signs.astype(np.int16) @ s_points).astype(np.int16)
        projection_active = projection_net >= self.config.projection_threshold

        association_net = (
            self.association_signs.astype(np.int16) @ projection_active.astype(np.int16)
        ).astype(np.int16)
        association_active = association_net >= self.config.association_threshold

        response_scores = np.zeros(self.config.n_responses, dtype=np.float64)
        for response in range(self.config.n_responses):
            active_in_source = association_active & self.response_sources[response]
            values = self.association_values[active_in_source]
            if values.size == 0:
                response_scores[response] = 0.0
            elif self.config.discrimination_mode == "sum":
                response_scores[response] = float(values.sum())
            else:
                response_scores[response] = float(values.mean())

        max_score = response_scores.max()
        tied = np.flatnonzero(np.isclose(response_scores, max_score))
        winner = int(self.rng.choice(tied))
        return ActivationTrace(
            projection_net=projection_net,
            projection_active=projection_active,
            association_net=association_net,
            association_active=association_active,
            response_scores=response_scores,
            winner=winner,
        )

    def predict_one(self, stimulus: NDArray[np.generic]) -> int:
        """Return the winning response index for one stimulus."""
        return self.forward(stimulus).winner

    def predict(self, stimuli: NDArray[np.generic]) -> NDArray[np.int64]:
        """Predict labels for a batch of flattened or image-shaped stimuli."""
        array = np.asarray(stimuli)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return np.asarray([self.predict_one(row) for row in array], dtype=np.int64)

    def _apply_gamma_update(
        self, response: int, association_active: BoolArray, direction: float
    ) -> None:
        """Apply a gain-conserving update within one response source-set.

        Active units change by direction * learning_rate. Inactive units within
        the same source-set compensate so the source-set's total value is
        conserved. This is a discrete research implementation of the gamma-like
        gain exchange described in the paper.
        """
        source = self.response_sources[response]
        active = source & association_active
        inactive = source & ~association_active
        n_active = int(active.sum())
        n_inactive = int(inactive.sum())
        if n_active == 0 or n_inactive == 0:
            return
        step = self.config.learning_rate
        self.association_values[active] += direction * step
        self.association_values[inactive] -= direction * step * n_active / n_inactive

    def train_step(self, stimulus: NDArray[np.generic], target: int) -> int:
        """Train on one stimulus and return the pre-update predicted response."""
        if not 0 <= int(target) < self.config.n_responses:
            raise ValueError("target is outside the configured response range")
        trace = self.forward(stimulus)
        predicted = trace.winner
        active = trace.association_active
        rule = self.config.learning_rule
        step = self.config.learning_rate

        if rule == "alpha":
            self.association_values[active & self.response_sources[int(target)]] += step
        elif rule == "beta":
            target_active = active & self.response_sources[int(target)]
            count = int(target_active.sum())
            if count:
                # Fixed total gain per reinforcement, distributed among active units.
                self.association_values[target_active] += step / count
        elif rule == "gamma":
            self._apply_gamma_update(int(target), active, direction=1.0)
        elif rule == "bivalent_gamma":
            for response in range(self.config.n_responses):
                direction = 1.0 if response == int(target) else -1.0
                self._apply_gamma_update(response, active, direction=direction)

        self.training_steps += 1
        return predicted

    def fit(
        self,
        stimuli: NDArray[np.generic],
        labels: NDArray[np.integer],
        epochs: int = 10,
        shuffle: bool = True,
    ) -> list[dict[str, float]]:
        """Train for multiple epochs and return per-epoch training metrics."""
        X = np.asarray(stimuli)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if len(X) != len(y):
            raise ValueError("stimuli and labels must have the same number of examples")
        if len(y) == 0:
            raise ValueError("training data cannot be empty")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(len(y)) if shuffle else np.arange(len(y))
            correct = 0
            for index in order:
                predicted = self.train_step(X[index], int(y[index]))
                correct += int(predicted == y[index])
            history.append({"epoch": float(epoch), "online_accuracy": correct / len(y)})
        return history

    def calculate_probabilities(self, stimuli: NDArray[np.generic]) -> dict[str, float]:
        """Direct calculation of the paper's P_a and P_e quantities."""
        X = np.asarray(stimuli)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        association_actives = []
        for row in X:
            trace = self.forward(row)
            association_actives.append(trace.association_active)
            
        acts = np.array(association_actives)
        # P_a: mean probability an association unit is active
        p_a = float(np.mean(acts))
        
        p_a_per_unit = np.mean(acts, axis=0)
        p_e_per_stimulus = np.mean(acts, axis=1)
        
        return {
            "P_a_mean": p_a,
            "P_a_std": float(np.std(p_a_per_unit)),
            "P_e_mean": float(np.mean(p_e_per_stimulus)),
            "P_e_std": float(np.std(p_e_per_stimulus)),
        }
