#! /bin/bash

#SBATCH --job-name=TOI-270d
#SBATCH --account=erausche1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --tasks-per-node=1
#SBATCH --mem=80G
#SBATCH --time=0-24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eraul@umich.edu
#SBATCH -o ./Slurm_output/TOI-270d.%j.out
#SBATCH -e ./Slurm_output/TOI-270d.%j.err
#SBATCH --get-user-env
#SBATCH --exclude=gl[3046,3059,3090,3100,3102,3143,3150,3153,3161,3174,3176,3188,3238,3278,3279,3319,3324,3341,3391,3406,3441]


declare -xr WDIR="/home/eraul/POSEIDON/run"

declare PATH=${PATH}:${WDIR}

time_start=`date '+%T%t%d_%h_06'`
  
echo ------------------------------------------------------
echo SBATCH: job name is $SLURM_JOB_NAME
echo SBATCH: job identifier is $SLURM_JOBID
echo SBATCH: sbatch is running on $SLURM_SUBMIT_HOST
echo SBATCH: executing queue is $SLURM_JOB_PARTITION
echo SBATCH: working directory is $SLURM_SUBMIT_DIR
echo SBATCH: node file is $SLURM_JOB_NODELIST
echo ------------------------------------------------------


cd ${WDIR}

mpirun -n 36 taskset -a -c 0-35 python -u TOI-270d_HCN_SO2_H2S_SOSS.py


time_end=`date '+%T%t%d_%h_06'`
echo Started at: $time_start
echo Ended at: $time_end
echo ------------------------------------------------------
echo Job ends
