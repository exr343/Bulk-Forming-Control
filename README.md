# Agility Forge

A differentiable, GPU-accelerated simulation of multi-hit rotational forging, built for a CIRP 2027 submission. This repo is being developed into a 3-stage pipeline:

1. **Data generation** (done) — `applications/Agility_Forge/generate_dataset.py` runs randomized multi-hit forging rollouts against the forging simulation and records the resulting state trajectories (`data/dataset_pretraining/`). `generate_square_rollout.py` adds scheduled, target-directed rollouts for finetuning (`data/dataset_finetuning/`): a 48-hit square-rod schedule adapted from a collaborator-provided industrial toolpath and scaled to this billet, plus jittered variants and a thin/bulge/head part.
2. **GNN surrogate model** (implemented and trained against the full 3D actuation space) — `applications/Agility_Forge/GNN/` learns a MeshGraphNets-style surrogate of the forging state from stage-1 rollouts. Die axial position, die orientation, *and* strike depth (`Uniform(0.5, 2.0)`mm, range grounded in the JAX-FORGE paper's reported forging depths) are all independently randomized per hit. A message-passing-depth sweep (1/5/15/45/135 steps) found accuracy saturates at ~5 steps with no benefit from going deeper. The earlier 2D-actuation version (strike depth held fixed at 2.5mm — later found to be an undocumented placeholder, not a validated value) is preserved for reference at `data/backup/dataset_2dim`/`GNN/runs_2dim/`. The 5-step model has since been finetuned on the square-rod rollouts (`GNN/finetune.py`), with pretraining data mixed in: over a full 48-hit autoregressive rollout of a held-out square run, final-hit Hausdorff error drops from 17.5mm to 2.1mm, with no loss of accuracy on the original test set. See [GNN/README.md](applications/Agility_Forge/GNN/README.md) for design decisions and results.
3. **MPC control loop** (implemented and validated against the real plant) — `applications/Agility_Forge/control/` uses the trained GNN surrogate as the fast predictive model inside a single-shooting SQP-BFGS receding-horizon controller, validating every planned control against the real simulation as the plant. Open-loop and closed-loop runs against held-out targets both land in the 0.15-0.27mm RMSE range. See [control/README.md](applications/Agility_Forge/control/README.md) for design decisions and validation results.

An earlier Koopman Autoencoder surrogate (`applications/Agility_Forge/koopman/`) was explored for stage 2; the GNN surrogate above is now the active approach.

## The forging simulation

The FEM core, `jax_forge/`, is a fork of [JAX-FEM](https://github.com/deepmodeling/jax-fem) (a differentiable finite element package built on [JAX](https://github.com/google/jax)) that's been extended for this project's constitutive model and solver. `applications/Agility_Forge/` builds a coupled thermo-mechanical (J2 finite-strain plasticity, temperature-dependent) simulation of a cylindrical billet under a rotating multi-hit press on top of it — see [applications/Agility_Forge/main.py](applications/Agility_Forge/main.py) for the driver and [CLAUDE.md](CLAUDE.md) for the full architecture.

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
