#!/bin/bash -l
#SBATCH --job-name=move_dataset
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x_%j.out
#
# Moves a dataset folder once the job writing it has ended (submit with
# --dependency=afterany:<writer job id>), so it's never moved mid-write.
# Usage (from the repo root):
#   sbatch --dependency=afterany:<job> applications/Agility_Forge/slurm_scripts/move_finished_dataset.sh <src> <dst>

set -euo pipefail
cd /home/exr343/CIRP_2027

src="$1"; dst="$2"
if [ ! -d "$src" ]; then echo "Source $src not found; nothing to move."; exit 0; fi
if [ -e "$dst" ]; then echo "Destination $dst already exists; refusing to overwrite." >&2; exit 1; fi
mkdir -p "$(dirname "$dst")"
mv "$src" "$dst"
echo "Moved $src -> $dst at $(date)"
