#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#SBATCH -c 7
#SBATCH --time=14-00:00:00
#SBATCH --job-name="E2W10fillin"
#SBATCH -p high2


EPS=1
MINLZ=100000
TLOW=0.3
W=10
E=2
DIM=3
NUM_REALIZATION=10
JOBNAME="E2W10-fillin"
FILENAME="${JOBNAME}.txt"



for i in "0.322 15" "0.322 28" "0.322 29" "0.322 30" "0.323 16" "0.323 27" "0.323 28"; do
    set -- $i # convert the "tuple" into the param args $1 $2...
    srun "/home/agoga/.conda/envs/fss/bin/python" localization_farm_numba.py $EPS $MINLZ $2 $W $TLOW $1 $E $DIM $NUM_REALIZATION $FILENAME &
done

wait
