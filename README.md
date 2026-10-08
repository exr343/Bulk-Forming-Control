# Agility Forge

A differentiable, GPU-accelerated simulation of multi-hit rotational forging, built for a CIRP 2027 submission. This repo is being developed into a 3-stage pipeline:

1. **Data generation** (done) — `applications/Agility_Forge/generate_dataset.py` runs randomized multi-hit forging rollouts against the forging simulation and records the resulting state trajectories (`data/dataset_pretraining/`). `generate_square_rollout.py` adds scheduled, target-directed rollouts for finetuning (`data/dataset_finetuning/`): a 48-hit square-rod schedule adapted from a collaborator-provided industrial toolpath and scaled to this billet, plus jittered variants and a thin/bulge/head part.
2. **GNN surrogate model** (implemented and trained against the full 3D actuation space) — `applications/Agility_Forge/GNN/` learns a MeshGraphNets-style surrogate of the forging state from stage-1 rollouts. Die axial position, die orientation, *and* strike depth (`Uniform(0.5, 2.0)`mm, range grounded in the JAX-FORGE paper's reported forging depths) are all independently randomized per hit. A message-passing-depth sweep (1/5/15/45/135 steps) found accuracy saturates at ~5 steps with no benefit from going deeper. The earlier 2D-actuation version (strike depth held fixed at 2.5mm — later found to be an undocumented placeholder, not a validated value) is preserved for reference at `data/backup/dataset_2dim`/`GNN/runs_2dim/`. The 5-step model has since been finetuned on the square-rod rollouts (`GNN/finetune.py`), with pretraining data mixed in: over a full 48-hit autoregressive rollout of a held-out square run, final-hit Hausdorff error drops from 17.5mm to 2.1mm, with no loss of accuracy on the original test set. See [GNN/README.md](applications/Agility_Forge/GNN/README.md) for design decisions and results.
3. **MPC control loop** (implemented and validated against the real plant) — `applications/Agility_Forge/control/` uses the trained GNN surrogate as the fast predictive model inside a single-shooting SQP-BFGS receding-horizon controller, validating every planned control against the real simulation as the plant. Open-loop and closed-loop runs against held-out targets both land in the 0.15-0.27mm RMSE range. See [control/README.md](applications/Agility_Forge/control/README.md) for design decisions and validation results.

An earlier Koopman Autoencoder surrogate (`applications/Agility_Forge/koopman/`) was explored for stage 2; the GNN surrogate above is now the active approach.

## The forging simulation

The FEM core, `jax_forge/`, is a fork of [JAX-FEM](https://github.com/deepmodeling/jax-fem) (a differentiable finite element package built on [JAX](https://github.com/google/jax)) that's been extended for this project's constitutive model and solver. `applications/Agility_Forge/` builds a coupled thermo-mechanical (J2 finite-strain plasticity, temperature-dependent) simulation of a cylindrical billet under a rotating multi-hit press on top of it — see [applications/Agility_Forge/main.py](applications/Agility_Forge/main.py) for the driver and [CLAUDE.md](CLAUDE.md) for the full architecture.

### The die

The die (added 2026-10-02) is set per hit in [hit_config.py](applications/Agility_Forge/hit_config.py) with `Hit(die_center_mm=..., compression_displacement=..., rotation_euler_x=...)`. Each hit works like this:

1. **Inputs:** the die's centre position along the bar (`die_center_mm`, a fixed spot in space in the mesh's x coordinate; the clamped end face is at x = −5 mm), the stroke (`compression_displacement`, how far each die moves in, mm) and the angle (`rotation_euler_x`, which direction the bar is squeezed, degrees).
2. **Two flat dies,** one on each side of the bar. Each is 12.7 mm wide along the bar, centred on the chosen spot, and wider than the bar in the other direction. The 12.7 mm width matches the flat tools in the JAX-FORGE paper (Sec. 3.3), which forged the same 15.9 mm stock as our billet.
3. **Each die starts touching the bar,** at the outermost point of the bar's surface on its side within the 12.7 mm footprint.
4. **Both dies move inward by the stroke** over the hit, so the bar is squeezed by twice the stroke in total.
5. **A surface point is pressed only if** it is currently within the 12.7 mm footprint and a die face has reached it. This is re-checked at every solver step. A pressed point stays on the die face and doesn't slide along it.
6. **The die directly moves only those pressed surface points.** Every other point, including the inside of the bar and any metal bulging out beside the die, is moved by the solver as the metal flows in response.
7. **The bar's end face at the clamp is held fixed** throughout.

**The original band, used for all data generated before 2026-10-02,** is still available for reproducing that data (`Hit(x_min_band=..., x_max_band=..., ...)`, fractions of the bar length). It differed in which points it pressed. It marked a 19.3 mm stretch of the *undeformed* bar and always pressed that same metal, wherever it had moved since. As the bar stretched, so did the pressed region: by the end of the 100-hit square runs, a 19.3 mm band covered about 34–35 mm of bar and had moved along with the metal, which no real die does. The new die stays where it is put and is always 12.7 mm wide.

Both are applied in [lib/boundary_conditions.py](applications/Agility_Forge/lib/boundary_conditions.py). The `jax_forge/` solver is unchanged. The data-generation and MPC scripts still use the original band.

### The reheat (agreed 2026-10-03)

**Today,** every hit starts with the bar's temperature reset to the starting profile, keyed to where each bit of metal was on the undeformed bar. The new reheat models the real process, where the bar is reheated in an induction coil (a solenoid) about every 6 hits:

1. **The cycle:** scan the bar's surface, reheat, then 6 hits. The MPC plans all 6 hits of a cycle from that scan.
2. **The coil** heats a 24.5 mm length of bar to 1096 °C. Outside the coil, the temperature drops by 5.45 °C per mm on both sides. Both numbers come from the starting profile, which is flat at 1096 °C from x = 72 mm to the free end (24.5 mm) and drops 5.45 °C/mm toward the clamp.
3. **The coil is centred** on the midpoint between the two outermost die positions of the 6 planned hits. Positions are where the bar is now, not where the metal started.
4. **A reheat only adds heat** (changed 2026-10-04): each spot keeps the hotter of its current temperature and the coil profile. The first rule, overwriting every spot with the profile, cooled the far end of long bars by up to ~300 °C at once, and the solver failed there. (The very first reheat, on the fresh billet, is the profile itself.)
5. **The bar is also reheated before hit 1,** with the coil centred on the first 6 hits, as a real bar comes straight out of the coil before its first hit. (The old starting profile is the same rule with the coil centred at 84.3 mm.)
6. **During the 6 hits,** cooling and the heat lost into the die stay exactly as they are today. The solver is not changed.
7. **The GNN gets temperature as a per-node input,** since the same shape and hit now give different results depending on how far into the cycle the hit is. Because the reheat is a formula of the bar's shape and the coil position, the MPC applies it directly; the GNN doesn't have to learn it.

Right after a reheat, metal is above 800 °C within 66.5 mm of the coil centre (12.25 mm of coil plus 54 mm of drop). Training the GNN on its accuracy over 6 hits in a row, matching how the MPC uses it, is under consideration.

The reheat is implemented in [control/plant_interface.py](applications/Agility_Forge/control/plant_interface.py) (`coil_temperature`, `ForgingPlant.reheat`, and `step(..., reheat=False)` to carry the temperature over between hits; the default `reheat=True` keeps the old reheat-every-hit behaviour). The first data on the new simulator is [generate_square_coil_rollout.py](applications/Agility_Forge/generate_square_coil_rollout.py): simple_square scaled to our bar (two passes, 6.27 mm then 5.3 mm half-gap, 0°/90° per station, 10.73 mm station step, no taper hit), with two more passes at 5.3 mm half-gap (two passes alone, 34 hits, left the bar about 13 mm thick and 122 mm long against the target's 10.6 mm and 153 mm), written to `data/dataset_die12_coil/`. The earlier runs made with the overwrite rule are in `data/backup/die12_coil_overwrite_reheat/`.

## Setup

```bash
conda env create -f environment.yml
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # required for petsc4py
```

On the HPC cluster this project runs on, see [cluster_setup.md](cluster_setup.md) for the SLURM/module-load workflow.

## Running

```bash
# Single multi-hit forging run (fixed hit schedule)
python -m applications.Agility_Forge.main

# Randomized hit schedule
python -m applications.Agility_Forge.main_random_hits

# Dataset generation for stages 2/3
python -m applications.Agility_Forge.generate_dataset --n-rollouts 10 --n-hits 10 --seed 42

# Scheduled square-rod rollout for finetuning (--jitter-seed N for a jittered variant, --part bulge_head for the thin/bulge/head part)
python -m applications.Agility_Forge.generate_square_rollout --dataset-dir applications/Agility_Forge/data/dataset_finetuning/square

# Finetune the GNN on the square-rod rollouts (with pretraining replay)
python -m applications.Agility_Forge.GNN.finetune
```

## License

`jax_forge/` is a derivative of JAX-FEM and this repo is licensed under the GNU General Public License v3 — see [LICENSE](LICENSE).

If you use the underlying FEM library, consider citing the original JAX-FEM paper:

```bibtex
@article{xue2023jax,
  title={JAX-FEM: A differentiable GPU-accelerated 3D finite element solver for automatic inverse design and mechanistic data science},
  author={Xue, Tianju and Liao, Shuheng and Gan, Zhengtao and Park, Chanwook and Xie, Xiaoyu and Liu, Wing Kam and Cao, Jian},
  journal={Computer Physics Communications},
  pages={108802},
  year={2023},
  publisher={Elsevier}
}
```
