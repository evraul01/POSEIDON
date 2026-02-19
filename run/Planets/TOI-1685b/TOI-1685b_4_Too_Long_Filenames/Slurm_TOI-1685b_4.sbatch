#! /bin/bash

#SBATCH --job-name=TOI-1685b_4
#SBATCH --account=ryanjmac0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --tasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=0-24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eraul@umich.edu
#SBATCH -o ./Slurm_output/TOI-1685b.%j.out
#SBATCH -e ./Slurm_output/TOI-1685b.%j.err
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

mpirun -n 24 taskset -a -c 0-35 python -u TOI-1685b_multigas-CH4_Gauss_TS_2000_joint_NIRISS_NIRSpec_Offset_Stacked_ExoTEP_R100.py


time_end=`date '+%T%t%d_%h_06'`
echo Started at: $time_start
echo Ended at: $time_end
echo ------------------------------------------------------
echo Job ends
