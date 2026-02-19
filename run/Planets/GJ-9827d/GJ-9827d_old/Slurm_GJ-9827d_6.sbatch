#! /bin/bash

#SBATCH --job-name=GJ-9827d_6
#SBATCH --account=erausche1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --tasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=0-24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eraul@umich.edu
#SBATCH -o ./Slurm_output/GJ-9827d.%j.out
#SBATCH -e ./Slurm_output/GJ-9827d.%j.err
#SBATCH --get-user-env
#SBATCH --exclude=gl[3036,3046,3056,3057,3059,3090,3094,3095,3100,3102,3110,3112,3122,3132,3141,3143,3145,3150,3169,3170,3174,3176,3186,3188,3190,3199,3219,3221,3238,3240,3243,3244,3261,3262,3267,3269,3273,3278,3279,3298,3303,3314,3317,3319,3324,3329,3338,3341,3346,3348,3353,3354,3377,3378,3391,3403,3405,3406,3407,3417,3441,3456]


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

mpirun -n 12 taskset -a -c 0-35 python -u GJ-9827d_NH3_SOSS_OS_NA_1000.py


time_end=`date '+%T%t%d_%h_06'`
echo Started at: $time_start
echo Ended at: $time_end
echo ------------------------------------------------------
echo Job ends
