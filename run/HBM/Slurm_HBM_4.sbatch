#! /bin/bash

#SBATCH --job-name=HBM_4
#SBATCH --account=ryanjmac0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --tasks-per-node=1
#SBATCH --mem=20G
#SBATCH --time=0-24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eraul@umich.edu
#SBATCH -o ./Slurm_output/HBM_forward.%j.out
#SBATCH -e ./Slurm_output/HBM_forward.%j.err
#SBATCH --get-user-env


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

mpirun -n 12 taskset -a -c 0-35 python -u HBM_forward_models_gaussian_retrievals.py


time_end=`date '+%T%t%d_%h_06'`
echo Started at: $time_start
echo Ended at: $time_end
echo ------------------------------------------------------
echo Job ends
