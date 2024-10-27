#!/bin/zsh
#SBATCH --nodes=2             
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --output=/home/ebi/slurm/ddp/project/output

export WORKON_HOME=~/.virtualenvs
source /usr/bin/virtualenvwrapper_lazy.sh
workon MY_ENV

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
srun python main.py --resize_size=32 --min_keep_ratio=0.3 --input_size=256 --epochs=1 --batch_size=160 --lr=1e-3 --mode=from_scratch --model_name=method3 --workers=2
