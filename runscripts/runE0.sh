#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -c 5
#SBATCH --time=14-00:00:00
#SBATCH --job-name="E010L30S1"
#SBATCH -p high2
#SBATCH --array=24-30:1


EPS=1
MINLZ=100000
TLOW=0.3
W=10
E=0
DIM=3
NUM_REALIZATION=10
JOBNAME="E0W10-L24-30-ccsearch1"
FILENAME="${JOBNAME}.txt"
L=$SLURM_ARRAY_TASK_ID

#clist=(0.271 0.284 0.293 0.296 0.296 0.299 0.302 0.304 0.307 0.313 0.323)
clist=(0.2731 0.28 0.292 0.304 .312)
#for C in "${clist[@]}"; do

echo "jobname=$JOBNAME eps=$EPS minlz=$MINLZ L=$L W=$W tLow=$TLOW tLBot=$TLBOT tLTop=$TLTOP tHBot=$THBOT tHTop=$THTOP c=$clist E=$E dim=$DIM num_realizations=$NUM_REALIZATION filename=$FILENAME" >> settings.txt

#clist=(0.319 0.312 0.307 0.305 0.302 0.300 0.298 0.296 0.292 0.285)
#for C in $(seq 0 0.1 1); do
for C in "${clist[@]}"; do
	srun "/home/agoga/.conda/envs/fss/bin/python" localization_farm_numba.py $EPS $MINLZ $L $W $TLOW $C $E $DIM $NUM_REALIZATION $FILENAME &
done

wait
