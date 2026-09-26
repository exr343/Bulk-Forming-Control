#!/bin/bash -l
#SBATCH --job-name=agility_forge_dataset_backfill
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=exr343@case.edu
#SBATCH --dependency=afterany:3794407:3795300
#
# Submit with:  sbatch slurm_scripts/submit_dataset_generation_backfill.sh
# (must be submitted with CWD at repo root)
#
# Runs automatically once BOTH the l40s (3794407, rollouts 201-250) and h100
# (3795300, rollouts 251-350) jobs have exited -- afterany, not afterok, so
# this still runs (and reports) even if one of them fails, rather than
# hanging forever.
#
# Scans every rollout_* dir in data/dataset_pretraining for a complete state sequence
# (undeformed.vtu + n_hits hit_XX_final.vtu files, per manifest.json's
# n_hits_per_rollout). Any rollout short of that -- including the known
# straggler #170 (stalled at 2/5 hits from the earlier pre-locking-fix
# manifest race) and any new stragglers left by 3794407/3795300 if either
# died mid-rollout -- doesn't need manual fixing: generate_dataset.py always
# appends at the next free index rather than resuming a specific one, and
# koopman/dataset.py's _group_complete_rollouts already filters incomplete
# rollouts out of training data automatically. So "backfill" here means:
# count the stragglers, then generate that many *replacement* rollouts to
# bring the complete-rollout count back up to the intended total. The
# straggler dirs themselves are left in place (harmless, already excluded
# from training).

set -euo pipefail

cd /home/exr343/CIRP_2027

module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

echo "Job started: $(date)"
echo "Host: $(hostname)"

echo "--- Parent job final states ---"
sacct -j 3794407,3795300 -o JobID,JobName,State,ExitCode -X

echo "--- Scanning data/dataset_pretraining for incomplete rollouts ---"
python3 - <<'PYEOF' 1>&2
import glob, os, json

dataset_dir = "applications/Agility_Forge/data/dataset_pretraining"
with open(os.path.join(dataset_dir, "manifest.json")) as f:
    meta = json.load(f)
n_hits = meta["n_hits_per_rollout"]

stragglers = []
for d in sorted(glob.glob(os.path.join(dataset_dir, "rollout_*"))):
    has_undeformed = os.path.exists(os.path.join(d, "undeformed.vtu"))
    n_hit_files = len(glob.glob(os.path.join(d, "hit_*_final.vtu")))
    if not has_undeformed or n_hit_files < n_hits:
        stragglers.append((os.path.basename(d), has_undeformed, n_hit_files))

print(f"Found {len(stragglers)} incomplete rollout(s):")
for name, has_undef, n in stragglers:
    print(f"  {name}: undeformed={has_undef} hit_final_count={n}/{n_hits}")

with open("/tmp/agility_forge_backfill_count.txt", "w") as f:
    f.write(str(len(stragglers)))
PYEOF
N_STRAGGLERS=$(cat /tmp/agility_forge_backfill_count.txt)

echo "--- Straggler count: $N_STRAGGLERS ---"

if [ "$N_STRAGGLERS" -gt 0 ]; then
    echo "Generating $N_STRAGGLERS replacement rollout(s) to restore the intended total"
    python -m applications.Agility_Forge.generate_dataset \
        --n-rollouts "$N_STRAGGLERS" \
        --n-hits 5 \
        --min-compression-displacement 0.5 \
        --max-compression-displacement 2.0
else
    echo "No incomplete rollouts found -- nothing to backfill."
fi

echo "Job finished: $(date)"
