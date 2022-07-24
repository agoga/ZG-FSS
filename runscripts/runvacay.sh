#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -c 10
#SBATCH --time=10-00:00:00
#SBATCH --job-name="L-29-30-5x"
#SBATCH -p high2
#SBATCH --array=29-30:1


EPS=1
MINLZ=100000
TLBOT=0
TLTOP=.6
THBOT=.4
THTOP=1
TLOW=0.3
W=10
E=2
DIM=3
NUM_REALIZATION=5
JOBNAME="E2W10-L29-30-5x"
FILENAME="${JOBNAME}.txt"
L=$SLURM_ARRAY_TASK_ID


echo "jobname=$JOBNAME eps=$EPS minlz=$MINLZ L=$L W=$W tLow=$TLOW tLBot=$TLBOT tLTop=$TLTOP tHBot=$THBOT tHTop=$THTOP c=$C E=$E dim=$DIM num_realizations=$NUM_REALIZATION filename=$FILENAME" >> settings.txt

clist=(0.308 0.306 0.3005 0.3003 0.3001 0.2997 0.2992 .2987 0.293 0.291)
#for C in $(seq 0 0.1 1); do
for C in "${clist[@]}"; do
	srun "/home/agoga/.conda/envs/fss/bin/python" localization_farm_numba.py $EPS $MINLZ $L $W $TLOW $C $E $DIM $NUM_REALIZATION $FILENAME &
done

wait
