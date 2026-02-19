#! /bin/bash

#SBATCH --job-name=GJ-3090b_12
#SBATCH --account=ryanjmac0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --tasks-per-node=1
#SBATCH --mem=40G
#SBATCH --time=0-24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eraul@umich.edu
#SBATCH -o ./Slurm_output/GJ-3090b.%j.out
#SBATCH -e ./Slurm_output/GJ-3090b.%j.err
#SBATCH --get-user-env
#SBATCH --exclude=gl[3031,3036,3042,3044,3046,3047,3053,3056,3057,3058,3059,3063,3065,3067,3072,3080,3089,3090,3092,3094,3095,3097,3100,3101,3102,3106,3109,3110,3111,3112,3114,3115,3120,3122,3123,3124,3131,3132,3133,3134,3139,3140,3141,3142,3143,3144,3145,3146,3150,3153,3155,3157,3159,3161,3163,3168,3169,3170,3172,3173,3174,3175,3176,3179,3184,3186,3188,3190,3191,3194,3199,3204,3205,3206,3213,3216,3217,3219,3220,3221,3223,3226,3228,3229,3233,3236,3237,3238,3239,3240,3242,3243,3244,3246,3248,3249,3254,3256,3261,3262,3264,3266,3267,3269,3270,3272,3273,3275,3276,3277,3278,3279,3280,3283,3284,3288,3290,3292,3293,3298,3299,3301,3303,3305,3310,3313,3314,3317,3318,3319,3320,3322,3323,3324,3329,3330,3332,3333,3334,3335,3338,3339,3341,3342,3346,3348,3350,3353,3354,3356,3357,3363,3370,3377,3378,3379,3390,3391,3395,3397,3403,3404,3405,3406,3407,3408,3413,3417,3418,3421,3429,3434,3436,3437,3439,3441,3448,3451,3453,3454,3455,3456,3479]


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

mpirun -n 12 taskset -a -c 0-35 python -u GJ-3090b_joint_multigas_CLR_TS_1000_Combined_NIRISS_R50R25_Combined_NIRSpec_R500.py


time_end=`date '+%T%t%d_%h_06'`
echo Started at: $time_start
echo Ended at: $time_end
echo ------------------------------------------------------
echo Job ends
