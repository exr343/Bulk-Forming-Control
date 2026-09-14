#!/bin/bash -l
#SBATCH --job-name=agility_forge_mp1_eval_timing
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Full test-set Chamfer/Hausdorff eval (merged into full_test_set_eval.json)
# plus clean gradient-timing benchmark, for the newly-trained M=1 model.

set -euo pipefail
cd /home/exr343/CIRP_2027
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

python -m applications.Agility_Forge.GNN.mp_sweep.eval_mp1_and_time
