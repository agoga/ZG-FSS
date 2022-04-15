#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -c 10
#SBATCH --time=10-00:00:00
#SBATCH --job-name="L-20-24-5x"
#SBATCH -p med2
#SBATCH --array=20-24:1


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
NUM_REALIZATION=10
JOBNAME="E2W10-L20-24-5x"
FILENAME="${JOBNAME}.txt"
L=$SLURM_ARRAY_TASK_ID

#clist=( 0.267 0.28 0.288 0.292 0.294 0.297 0.3 0.302 0.306 0.314 0.327)
#for C in "${clist[@]}"; do

echo "jobname=$JOBNAME eps=$EPS minlz=$MINLZ L=$L W=$W tLow=$TLOW tLBot=$TLBOT tLTop=$TLTOP tHBot=$THBOT tHTop=$THTOP c=$C E=$E dim=$DIM num_realizations=$NUM_REALIZATION filename=$FILENAME" >> settings.txt

clist=(0.285 0.292 0.297 0.299 0.3 0.302 0.305 0.307 0.312 0.319)
#for C in $(seq 0 0.1 1); do
for C in "${clist[@]}"; do
	srun "/home/agoga/.conda/envs/fss/bin/python" localization_farm_numba.py $EPS $MINLZ $L $W $TLOW $C $E $DIM $NUM_REALIZATION $FILENAME &
done

wait
