#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name Mo16eg
#SBATCH --account=2016234
#SBATCH --partition h100
#SBATCH --time 48:00:00
#SBATCH --exclusive
##SBATCH --mail-user=youremail@unm.edu

# On Xena:
module load abinit/10.4.5

jobstart=`date`
srun --mpi=pmi2 -n $SLURM_NTASKS abinit trf2_1.abi > trf2_1.log 2> trf2_1.err
jobend=`date`
echo "  abinit job start: $jobstart"
echo $SLURM_NTASKS
echo $SLURM_JOB_NODELIST
echo "  abinit job end:   $jobend"
