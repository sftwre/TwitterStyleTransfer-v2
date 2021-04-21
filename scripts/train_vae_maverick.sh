#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede2 KNL nodes
#----------------------------------------------------

#SBATCH -J vae_job_$(date +%Y%m%d_%H%M%S)           # Job name
#SBATCH -o myjob.o%j       # Name of stdout output file
#SBATCH -e myjob.e%j       # Name of stderr error file
#SBATCH -p normal          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=isaac.buitrago@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A TwitterStyleTransfer       # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...

# Launch serial code...

while getopts epochs:lr: flag
do
    case "${flag}" in
        epochs) epochs=${OPTARG};;
        lr) lr=${OPTARG};;
    esac
done

python3 ../trainer/train_vae.py --epochs $epochs --lr $lr

# ---------------------------------------------------