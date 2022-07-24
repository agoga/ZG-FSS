#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -c 10
#SBATCH --time=10-00:00:00
#SBATCH --job-name="W12-10-24-10x"
#SBATCH -p high2
#SBATCH --array=10-24:1


EPS=1
MINLZ=100000
TLBOT=0
TLTOP=.6
THBOT=.4
THTOP=1
TLOW=0.3
W=12
E=2
DIM=3
NUM_REALIZATION=5
JOBNAME="E2W12-L10-24-10x"
FILENAME="${JOBNAME}.txt"
L=$SLURM_ARRAY_TASK_ID



echo "jobname=$JOBNAME eps=$EPS minlz=$MINLZ L=$L W=$W tLow=$TLOW tLBot=$TLBOT tLTop=$TLTOP tHBot=$THBOT tHTop=$THTOP c=$C E=$E dim=$DIM num_realizations=$NUM_REALIZATION filename=$FILENAME" >> settings.txt

#clist=(0.319 0.312 0.307 0.305 0.302 0.300 0.298 0.296 0.292 0.285)
clist=(0.41 0.411 0.421 0.431 0.441 0.451 0.461 0.471 0.481 0.491)

for C in "${clist[@]}"; do
	srun "/home/agoga/.conda/envs/fss/bin/python" localization_farm_numba.py $EPS $MINLZ $L $W $TLOW $C $E $DIM $NUM_REALIZATION $FILENAME &
done

wait

#clist=( 0.267 0.28 0.288 0.292 0.294 0.297 0.3 0.302 0.306 0.314 0.327)
#for C in "${clist[@]}"; do
#for C in $(seq 0 0.1 1); do
