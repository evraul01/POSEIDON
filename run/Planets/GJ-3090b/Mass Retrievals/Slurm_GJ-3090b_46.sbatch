#! /bin/bash

#SBATCH --job-name=GJ-3090b_46
#SBATCH --account=ryanjmac1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --tasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time=0-24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eraul@umich.edu
#SBATCH -o ./Slurm_output/GJ-3090b.%j.out
#SBATCH -e ./Slurm_output/GJ-3090b.%j.err
#SBATCH --get-user-env
#SBATCH --exclude=gl[3036,3046,3047,3056,3057,3058,3059,3063,3065,3067,3072,3078,3080,3089,3090,3092,3093,3094,3095,3097,3100,3101,3102,3110,3111,3112,3114,3115,3122,3123,3130,3132,3133,3139,3140,3141,3142,3143,3144,3145,3146,3150,3153,3157,3161,3168,3169,3170,3173,3174,3175,3176,3186,3188,3190,3194,3199,3219,3220,3221,3226,3238,3239,3240,3241,3243,3244,3248,3261,3262,3267,3269,3272,3273,3276,3278,3279,3284,3285,3292,3293,3298,3303,3314,3317,3318,3319,3320,3323,3324,3332,3333,3334,3335,3338,3339,3341,3342,3346,3348,3353,3354,3356,3377,3378,3391,3395,3397,3403,3405,3406,3407,3417,3421,3437,3441,3456]


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

mpirun -n 12 taskset -a -c 0-35 python -u GJ_3090b_joint_multigas_CLR_TS_200_NIRISS_Visit1_G395H.py


time_end=`date '+%T%t%d_%h_06'`
echo Started at: $time_start
echo Ended at: $time_end
echo ------------------------------------------------------
echo Job ends
