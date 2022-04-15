#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -c 12
#SBATCH --time=10-00:00:00
#SBATCH --job-name="FSS_plot"
#SBATCH -p high2



CPUS=12
MINL=10
MAXL=24
MINC=.20
MAXC=.4
RESAMPLES=12
JOBNAME='FSS_PLOT'
FILENAME="${JOBNAME}.txt"
DATAFILE="E2W10Lz100K.csv"

#clist=( 0.267 0.28 0.288 0.292 0.294 0.297 0.3 0.302 0.306 0.314 0.327)
#for C in "${clist[@]}"; do

#echo "jobname=$JOBNAME eps=$EPS minlz=$MINLZ L=$L W=$W tLow=$TLOW tLBot=$TLBOT tLTop=$TLTOP tHBot=$THBOT tHTop=$THTOP c=$C E=$E dim=$DIM num_realizations=$NUM_REALIZATION filename=$FILENAME" >> settings.txt

#clist=(0.281 0.285 0.292 0.297 0.299 0.302 0.305 0.307 0.312 0.319 0.322)
#for C in $(seq 0 0.1 1); do
#for C in "${clist[@]}"; do
#llist=(2)
srun "/home/agoga/.conda/envs/fss/bin/python" kindis_FSS.py $DATAFILE $MINL $MAXL $MINC $MAXC $RESAMPLES $CPUS
#done

wait
