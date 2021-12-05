#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -N 1
#SBATCH -n 11
#SBATCH --time=7-00:00:00
#SBATCH --job-name="E0W10box22"
#SBATCH -p high2
#SBATCH --array=8-11:1


EPS=1
MINLZ=500000
TLBOT=0
TLTOP=.3
THBOT=1
THTOP=1
W=10
E=0
DIM=3
AVG=1
FILENAME="E0W10boxtest_tHi1.txt"
L=$SLURM_ARRAY_TASK_ID

for C in $(seq 0 0.1 1); do
	srun --exclusive --ntasks 1 python localization_farm_4.py $EPS $MINLZ $L $W $TLBOT $TLTOP $THBOT $THTOP $C $E $DIM $AVG $FILENAME & 	
done
wait
