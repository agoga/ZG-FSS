#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -N 1-2
#SBATCH -n 11
#SBATCH --time=10-00:00:00
#SBATCH --job-name="E2W10wide2"
#SBATCH -p high2
#SBATCH --array=14-28:1


EPS=1
MINLZ=100000
TLOW=0.3
W=10
E=2
DIM=3
NUM_MEAS=1
FILENAME="offdiagE2W10wide2.txt"
L=$SLURM_ARRAY_TASK_ID

clist=( 0.271 0.284 0.291 0.296 0.298 0.301 0.304 0.306 0.311 0.318 0.331)
for C in "${clist[@]}"; do
	srun --exclusive --ntasks 1 python localization_farm.py $EPS $MINLZ $L $W $TLOW $C $E $DIM $NUM_MEAS $FILENAME & 	
done
wait
