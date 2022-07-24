#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -c 10
#SBATCH --time=10-00:00:00
#SBATCH --job-name="LZTEST"
#SBATCH -p med2
#SBATCH --array=0-5:1


EPS=1
LZVALUES=({50000,75000,100000,125000,175000,200000})
MINLZ=${LZVALUES[$SLURM_ARRAY_TASK_ID]}
TLBOT=0
TLTOP=.6
THBOT=.4
THTOP=1
TLOW=0.3
W=10
E=2
DIM=3
NUM_REALIZATION=1
JOBNAME="LZTEST"
FILENAME="${JOBNAME}.txt"
L=28 #$SLURM_ARRAY_TASK_ID



echo "jobname=$JOBNAME eps=$EPS minlz=$MINLZ L=$L W=$W tLow=$TLOW tLBot=$TLBOT tLTop=$TLTOP tHBot=$THBOT tHTop=$THTOP c=$C E=$E dim=$DIM num_realizations=$NUM_REALIZATION filename=$FILENAME" >> settings.txt

#clist=(0.309 0.307 0.305 0.302 0.3002 0.3001 0.2999 0.297 0.293 0.291)
#clist=(0.41 0.411 0.421 0.431 0.441 0.451 0.461 0.471 0.481 0.491)
clist=(0.267 0.28 0.288 0.292 0.294 0.297 0.3 0.302 0.306 0.314 0.327)

for C in "${clist[@]}"; do
	srun "/home/agoga/.conda/envs/fss/bin/python" localization_farm_numba.py $EPS $MINLZ $L $W $TLOW $C $E $DIM $NUM_REALIZATION $FILENAME &
done

wait

#clist=( 0.267 0.28 0.288 0.292 0.294 0.297 0.3 0.302 0.306 0.314 0.327)
#for C in "${clist[@]}"; do
#for C in $(seq 0 0.1 1); do
