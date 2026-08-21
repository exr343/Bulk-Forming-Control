## Setup in Cluster
1. ssh into the cluster
2. Request a GPU:

```bash
srun -p gpu --gres=gpu:1 -c 4 --mem=32G -t 04:00:00 --pty bash -l
```

3. cd to the repo (e.g. `/home/exr343/CIRP_2027`)
4. Activate the environment (automatic if these lines are in `~/.bashrc`, otherwise run manually, in this order):

```bash
module load Miniconda3/23.10.0-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate jax-fem-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
## Run the experiment 
5. Run the application:

```bash
python -m applications.Agility_Forge.main
```
