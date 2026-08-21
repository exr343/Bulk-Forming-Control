"""MPC loop: Koopman Autoencoder for prediction, ForgingPlant for ground truth.

Receding-horizon control over a sequence of forging hits: at each step, plan a
control (or short sequence of controls) by optimizing the trained Koopman
model's latent-linear predictions, apply the first control to the real plant
(applications.Agility_Forge.control.plant_interface.ForgingPlant) to get the
true next state, then re-plan from that true state.

TODO: this module is a stub. See ./README.md for the open solver/horizon
design questions.
"""

import numpy as onp
import torch

from applications.Agility_Forge.koopman.model import KoopmanAutoencoder
from applications.Agility_Forge.control.plant_interface import ForgingPlant, ForgingState


class MPCController:
    def __init__(self, koopman_model: KoopmanAutoencoder, plant: ForgingPlant, horizon: int, target_state):
        self.koopman_model = koopman_model
        self.plant = plant
        self.horizon = horizon
        self.target_state = target_state
        raise NotImplementedError("TODO: store cost weights, control bounds, solver config")

    def plan(self, state: ForgingState):
        """Optimize a control sequence over `self.horizon` steps in the Koopman
        latent space and return the first control to apply.

        TODO: with linear latent dynamics (z_{t+1} = A z_t + B u_t) and a
        quadratic tracking cost, this reduces to a QP — solve with e.g. cvxpy
        or a hand-rolled projected-gradient loop in torch. Solver choice is an
        open question (see ./README.md).
        """
        raise NotImplementedError

    def run(self, n_steps: int):
        """Receding-horizon loop: plan -> apply to real plant -> observe -> replan.

        TODO: at each step, call self.plan(state) for the next control, apply it
        via self.plant.step(state, control) to get the *true* next state (not
        the Koopman model's prediction), and log the plan-vs-actual mismatch —
        that mismatch is exactly what validates (or invalidates) the surrogate.
        """
        raise NotImplementedError
