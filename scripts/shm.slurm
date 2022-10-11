#!/bin/bash
#SBATCH -J shm           # Job name
#SBATCH -o log/shm.o%j       # Name of stdout output file
#SBATCH -e log/shm.e%j       # Name of stderr error file
#SBATCH -p gpu-a100            # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 3
#SBATCH -t 00:00:03        # Run time (hh:mm:ss)
#SBATCH -A Deep-Learning-at-Sca       # Allocation name (req'd if you have more than 1)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=sli@tacc.utexas.edu


df -h