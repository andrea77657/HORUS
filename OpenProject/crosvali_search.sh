#!/bin/bash
#SBATCH -n 1
#SBATCH --mem-per-cpu=32g
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --array=0-29
#SBATCH --job-name=gridsearch
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt

module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1          
module load python/3.10.13
module load eth_proxy       

cd "$SLURM_SUBMIT_DIR"

# Define grid search parameters
lrs=(0.001 0.001 0.001 0.001 0.001 0.001 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005
     0.0003 0.0003 0.0003 0.0003 0.0003 0.0003 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001
     0.00005 0.00005 0.00005 0.00005 0.00005 0.00005)
bss=(64 32 16 64 32 16 64 32 16 64 32 16 64 32 16 64 32 16 64 32 16 64 32 16 64 32 16 64 32 16)
pats=(5 5 5 10 10 10 5 5 5 10 10 10 5 5 5 10 10 10 5 5 5 10 10 10 5 5 5 10 10 10)
folds=(1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5)

# Select config for this array index
i=$SLURM_ARRAY_TASK_ID
lr=${lrs[$i]}
bs=${bss[$i]}
pat=${pats[$i]}
fold=${folds[$i]}

echo "Running config: LR=$lr, BS=$bs, PATIENCE=$pat, FOLD=$fold"

python3 -u harderimagesKfold.py $lr $bs $pat $fold
