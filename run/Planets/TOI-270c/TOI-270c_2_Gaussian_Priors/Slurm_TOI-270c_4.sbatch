#! /bin/bash

#SBATCH --job-name=TOI-270c_4
#SBATCH --account=mrmeyer0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --tasks-per-node=1
#SBATCH --mem=75G
#SBATCH --time=0-18:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eraul@umich.edu
#SBATCH -o ./Slurm_output/TOI-270c.%j.out
#SBATCH -e ./Slurm_output/TOI-270c.%j.err
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

mpirun -n 20 taskset -a -c 0-35 python -u TOI-270c_multigas_Gauss-T_2000_Offset_NIRSpec_G395H_Eureka.py


time_end=`date '+%T%t%d_%h_06'`
echo Started at: $time_start
echo Ended at: $time_end
echo ------------------------------------------------------
echo Job ends
