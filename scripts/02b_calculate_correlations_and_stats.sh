#!/bin/bash
#SBATCH -J TIL_correlations
#SBATCH -A MENON-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --array=0-999
#SBATCH --mail-type=NONE
#SBATCH --output=/home/sb2406/CENTER-TBI_TIL/bootstrapping_results/hpc_logs/TIL_correlations_trial_%a.out

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 02b_calculate_correlations_and_stats.py $SLURM_ARRAY_TASK_ID