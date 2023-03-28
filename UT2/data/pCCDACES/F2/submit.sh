#!/bin/bash
#SBATCH --job-name=a11IPCor
#SBATCH --output=zzz.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=2gb

echo $SLURM_NODELIST
echo $SLURM_JOBID
export PATH=/blue/bartlett/z.windom/SDACES/ACESII-2.14.0/bin:$PATH
export TESTROOT=$SLURM_SUBMIT_DIR

echo "`date`" > out.out

xrunpccd >> out.out
echo "`date`" >> out.out

