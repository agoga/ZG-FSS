#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -N 1
#SBATCH -n 11
#SBATCH --time=7-00:00:00
#SBATCH --job-name="tLtH_06_04"
#SBATCH -p high2
#SBATCH --array=6-14:1


EPS=1
MINLZ=500000
TLBOT=0
TLTOP=.6
THBOT=.4
THTOP=1
W=10
E=0
DIM=3
AVG=1
FILENAME="E0W10uniform_tLtH_0406.txt"
L=$SLURM_ARRAY_TASK_ID

echo "eps=$EPS minlz=$MINLZ L=$L W=$W tLBot=$TLBOT tLTop=$TLTOP tHBot=$THBOT tHTop=$THTOP c=$C E=$E dim=$DIM avg=$AVG filename=$FILENAME" > settings.txt

for C in $(seq 0 0.1 1); do
	srun --exclusive --ntasks 1 python localization_farm_4.py $EPS $MINLZ $L $W $TLBOT $TLTOP $THBOT $THTOP $C $E $DIM $AVG $FILENAME & 	
done

wait
