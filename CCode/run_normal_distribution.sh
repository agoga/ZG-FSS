#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --time=7-00:00:00
#SBATCH --job-name="E0W10norm-06-2"
#SBATCH -p high2
#SBATCH --array=6-14:1


EPS=1
MINLZ=500000
#TH TL unused in normal dsitribution
TLBOT=1 #t-low parameter 1
TLTOP=0.1 #t-low parameter 2
THBOT=1 #t-high parameter 1
THTOP=0.1 #t-high parameter 2
SIG=.03
C=.99
W=12
E=4
DIM=3
AVG=1
FILENAME="normtestE4W12.txt"
JOBNAME="normtestE4W12"
L=$SLURM_ARRAY_TASK_ID

echo "eps=$EPS minlz=$MINLZ L=$L W=$W tLBot=$TLBOT tLTop=$TLTOP tHBot=$THBOT tHTop=$THTOP c=$C E=$E dim=$DIM avg=$AVG filename=$FILENAME" > settings.txt

for CEN in $(seq 0.1 0.1 1); do
	srun --job-name $JOBNAME --exclusive --ntasks 1 python localization_farm.py $EPS $MINLZ $L $W $CEN $SIG $CEN $SIG $C $E $DIM $AVG $FILENAME & 	
done

wait
