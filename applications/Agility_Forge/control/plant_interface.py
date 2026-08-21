"""Thin wrapper exposing the Agility_Forge forging simulation as an MPC "plant".

Wraps applications.Agility_Forge.lib.constitutive.ThermalMechanical +
applications.Agility_Forge.lib.time_stepper.AutomaticTimeStepperTM (the same
pieces applications/Agility_Forge/main.py drives directly) behind a
step(state, control) -> next_state interface, decoupled from the Koopman model
so the MPC loop (./mpc.py) can call either interchangeably.

TODO: this module is a stub.
"""

import numpy as onp

from applications.Agility_Forge.hit_config import Hit
from applications.Agility_Forge.lib.constitutive import ThermalMechanical
from applications.Agility_Forge.lib.time_stepper import AutomaticTimeStepperTM
from applications.Agility_Forge.lib.boundary_conditions import (
    build_cylinder_press_bcs,
    refresh_problem_surface_integrals,
)


class ForgingState:
    """Plain-NumPy snapshot of the plant state: (sol_u, sol_dT, int_vars).

    Kept as NumPy (not JAX or Torch arrays) at this boundary so the Koopman
    model (PyTorch) and the plant (JAX) never need to convert tensors
    directly between frameworks.
    """

    def __init__(self, sol_u, sol_dT, int_vars):
        self.sol_u = sol_u
        self.sol_dT = sol_dT
        self.int_vars = int_vars


class ForgingPlant:
    """MPC-facing wrapper around one shared ThermalMechanical problem instance.

    TODO: mesh/geometry setup mirrors applications/Agility_Forge/main.py's
    run_thermo_mech_cylinder_press_multi_hits() — factor the mesh-loading and
    initial-temperature-profile logic there into a shared helper this class
    can call, rather than duplicating it.
    """

    def __init__(self, mesh, R, H, T_linear_fn):
        raise NotImplementedError(
            "TODO: build the shared ThermalMechanical problem (see main.py's "
            "'Build problem once' section) and store mesh/R/H/T_linear_fn"
        )

    def reset(self) -> ForgingState:
        """Return the plant to its undeformed, initial-temperature state."""
        raise NotImplementedError

    def step(self, state: ForgingState, hit: Hit) -> ForgingState:
        """Apply one forging hit and return the resulting state.

        TODO: build_cylinder_press_bcs(...) + refresh_problem_surface_integrals(...)
        + AutomaticTimeStepperTM(...).run(...), following the per-hit loop body
        in main.py's run_thermo_mech_cylinder_press_multi_hits() (minus the
        inter-hit thermal-relaxation ramp, which is a modeling choice for
        whoever calls this — MPC may want it, or may want raw back-to-back hits).
        Must raise (or return an explicit failure) if the stepper fails to
        converge, so the MPC loop can react (e.g. reject that control and
        re-plan) rather than silently propagating a bad state.
        """
        raise NotImplementedError
