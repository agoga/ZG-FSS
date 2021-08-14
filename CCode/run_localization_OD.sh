#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -N 1
#SBATCH -n 21
#SBATCH --time=7-00:00:00
#SBATCH --job-name="E6W15"
#SBATCH -p high2
#SBATCH --array=6-16:1


EPS=1
MINLZ=1000000
TLOW=0.3
W=15
E=6
DIM=3
AVG=1
FILENAME="offdiagE6W15.txt"
L=$SLURM_ARRAY_TASK_ID

for C in $(seq 0.7 0.01 0.9); do
	srun --exclusive --ntasks 1 python localization_farm.py $EPS $MINLZ $L $W $TLOW $C $E $DIM $AVG $FILENAME & 	
done
wait